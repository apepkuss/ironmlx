// Build: swiftc -parse-as-library examples/speech-client.swift -o /tmp/ironmlx-speech-client
// macOS example: complete WAV or bounded, incremental PCM playback via AVAudioEngine.
import AVFoundation
import Foundation
import Darwin

struct ClientError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

struct PCMDecoder {
    private var pending: UInt8?
    mutating func decode(_ bytes: [UInt8]) -> [Float] {
        var samples: [Float] = []
        samples.reserveCapacity((bytes.count + 1) / 2)
        for byte in bytes {
            if let low = pending {
                samples.append(Float(Int16(bitPattern: UInt16(low) | UInt16(byte) << 8)) / 32768)
                pending = nil
            } else {
                pending = byte
            }
        }
        return samples
    }
    func finish() throws {
        if pending != nil { throw ClientError("Truncated PCM: unmatched final byte") }
    }
}

struct Options {
    var endpoint = "http://127.0.0.1:9068/v1/audio/speech"
    var model = "mlx-community/IndexTTS-2.5-fp16"
    var text = ""
    var reference = ""
    var format = "pcm"
    var output: String?
    var cancelAfterMS: UInt64?
    init(_ args: [String]) throws {
        var i = 0
        while i < args.count {
            guard i + 1 < args.count else { throw ClientError("Missing value for \(args[i])") }
            let value = args[i + 1]
            switch args[i] {
            case "--url": endpoint = value
            case "--model": model = value
            case "--text": text = value
            case "--reference": reference = value
            case "--format": format = value
            case "--output": output = value
            case "--cancel-after-ms":
                guard let delay = UInt64(value), delay > 0, delay <= 1_200_000 else {
                    throw ClientError("Invalid cancellation delay")
                }
                cancelAfterMS = delay
            default: throw ClientError("Unknown option \(args[i])")
            }
            i += 2
        }
        guard !text.isEmpty, !reference.isEmpty, ["wav", "pcm"].contains(format),
              let url = URL(string: endpoint), ["http", "https"].contains(url.scheme) else {
            throw ClientError("Use --reference FILE --text TEXT [--format wav|pcm] [--url URL] [--model ID] [--output FILE] [--cancel-after-ms N]")
        }
    }
}

@MainActor
final class Playback {
    let engine = AVAudioEngine()
    let node = AVAudioPlayerNode()
    let format = AVAudioFormat(standardFormatWithSampleRate: 22050, channels: 1)!
    private(set) var queuedFrames = 0
    private(set) var playedFrames = 0
    private(set) var firstPlayedMS: Double?
    private(set) var peakQueuedFrames = 0
    let start: ContinuousClock.Instant
    init(start: ContinuousClock.Instant) throws {
        self.start = start
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        try engine.start()
        node.play()
    }
    func append(_ samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        guard samples.count <= 8192 else { throw ClientError("Playback block too large") }
        let waitStart = ContinuousClock.now
        while queuedFrames + samples.count > 44100 {
            try await waitForDevice(since: waitStart)
        }
        try Task.checkCancellation()
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { throw ClientError("Audio buffer allocation failed") }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        queuedFrames += samples.count
        peakQueuedFrames = max(peakQueuedFrames, queuedFrames)
        let count = samples.count
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.queuedFrames -= count
                self.playedFrames += count
                if self.firstPlayedMS == nil { self.firstPlayedMS = elapsedMS(self.start) }
            }
        }
    }
    func finish() async throws {
        let waitStart = ContinuousClock.now
        while queuedFrames > 0 { try await waitForDevice(since: waitStart) }
    }
    private func waitForDevice(since start: ContinuousClock.Instant) async throws {
        if start.duration(to: .now) > .seconds(10) || !engine.isRunning {
            throw ClientError("Audio device stopped or playback callbacks stalled")
        }
        try await Task.sleep(for: .milliseconds(5))
    }
    func stop() { node.stop(); engine.stop() }
}

func elapsedMS(_ start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: .now).components
    return Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15
}

func validateHeaders(_ response: HTTPURLResponse, format: String) throws {
    for (key, value) in [
        "Content-Type": format == "pcm" ? "audio/pcm" : "audio/wav",
        "X-Audio-Sample-Rate": "22050", "X-Audio-Channels": "1",
        "X-Audio-Sample-Format": "s16le",
    ] {
        guard response.value(forHTTPHeaderField: key) == value else {
            throw ClientError("Unexpected \(key); refusing to play with guessed audio format")
        }
    }
    if format == "pcm" {
        guard response.value(forHTTPHeaderField: "X-IronMLX-Streaming-Granularity") == "segment",
              response.value(forHTTPHeaderField: "Transfer-Encoding")?.lowercased() == "chunked",
              response.value(forHTTPHeaderField: "Content-Length") == nil else {
            throw ClientError("Unexpected PCM streaming headers")
        }
    }
}

func wavPCM(_ data: [UInt8]) throws -> [UInt8] {
    func u16(_ i: Int) -> UInt16 { UInt16(data[i]) | UInt16(data[i + 1]) << 8 }
    func u32(_ i: Int) -> UInt32 { UInt32(u16(i)) | UInt32(u16(i + 2)) << 16 }
    guard data.count >= 44 else { throw ClientError("Truncated WAV") }
    guard String(bytes: data[0..<4], encoding: .ascii) == "RIFF",
          String(bytes: data[8..<16], encoding: .ascii) == "WAVEfmt ",
          String(bytes: data[36..<40], encoding: .ascii) == "data",
          u32(4) == data.count - 8, u32(16) == 16,
          u16(20) == 1, u16(22) == 1, u32(24) == 22050,
          u32(28) == 44100, u16(32) == 2, u16(34) == 16,
          u32(40) == data.count - 44, (data.count - 44) % 2 == 0 else {
        throw ClientError("Invalid WAV layout or audio format")
    }
    return Array(data.dropFirst(44))
}

// CFNetwork can accept EOF without the terminating HTTP/1.1 chunk. Use the
// system curl transport, which verifies framing and returns nonzero on truncation.
// A pipe bounds unread bytes; waiting for the audio device also backpressures HTTP.
@MainActor
final class Transport {
    let process = Process()
    let pipe = Pipe()
    let root: URL
    let errors: FileHandle
    init(options: Options, body: Data) throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("ironmlx-speech-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        do {
            try body.write(to: root.appendingPathComponent("request.json"))
            var headers = "Content-Type: application/json\n"
            if let key = ProcessInfo.processInfo.environment["IRONMLX_API_KEY"] {
                guard !key.contains("\r"), !key.contains("\n") else { throw ClientError("Invalid API key") }
                headers += "Authorization: Bearer \(key)\n"
            }
            try Data(headers.utf8).write(to: root.appendingPathComponent("request.headers"))
            let errorURL = root.appendingPathComponent("stderr")
            guard FileManager.default.createFile(atPath: errorURL.path, contents: nil) else {
                throw ClientError("Cannot create transport log")
            }
            errors = try FileHandle(forWritingTo: errorURL)
            process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
            process.arguments = ["--disable", "--silent", "--show-error", "--no-buffer", "--http1.1",
                "--noproxy", "localhost,127.0.0.1,::1", "--connect-timeout", "10", "--max-time", "1200",
                "--speed-limit", "1", "--speed-time", "330", "--request", "POST",
                "--header", "@" + root.appendingPathComponent("request.headers").path,
                "--data-binary", "@" + root.appendingPathComponent("request.json").path,
                "--dump-header", root.appendingPathComponent("response.headers").path,
                "--output", "-", "--url", options.endpoint]
            process.standardOutput = pipe
            process.standardError = errors
            try process.run()
            try pipe.fileHandleForWriting.close()
        } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
    }
    func read() async throws -> Data {
        let handle = pipe.fileHandleForReading
        return try await withTaskCancellationHandler {
            let data = try await Task.detached { try handle.read(upToCount: 777) ?? Data() }.value
            try Task.checkCancellation()
            return data
        } onCancel: {
            Task { @MainActor in self.stop() }
        }
    }
    func response(url: URL) throws -> HTTPURLResponse {
        let text = try String(contentsOf: root.appendingPathComponent("response.headers"), encoding: .utf8)
        // Ignore informational/proxy CONNECT blocks, retaining the final response.
        let blocks = text.components(separatedBy: "\r\n\r\n").filter { $0.hasPrefix("HTTP/") }
        guard let block = blocks.last else { throw ClientError("Missing HTTP headers") }
        let lines = block.components(separatedBy: "\r\n")
        guard let status = lines[0].split(separator: " ").dropFirst().first.flatMap({ Int($0) }) else {
            throw ClientError("Invalid HTTP status")
        }
        var fields: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { throw ClientError("Invalid HTTP header") }
            let key = String(line[..<colon]).lowercased()
            guard fields[key] == nil else { throw ClientError("Duplicate HTTP header") }
            fields[key] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        guard let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: fields) else {
            throw ClientError("Invalid HTTP response")
        }
        return response
    }
    func finish() async throws {
        while process.isRunning { try await Task.sleep(for: .milliseconds(5)) }
        try Task.checkCancellation()
        guard process.terminationReason == .exit, process.terminationStatus == 0 else {
            let message = (try? String(contentsOf: root.appendingPathComponent("stderr"), encoding: .utf8)) ?? ""
            throw ClientError("Transport failed (curl \(process.terminationStatus)): \(message)")
        }
    }
    func stop() { if process.isRunning { process.terminate() } }
    func close() {
        stop()
        process.waitUntilExit()
        try? pipe.fileHandleForReading.close()
        try? errors.close()
        try? FileManager.default.removeItem(at: root)
    }
}

@MainActor
func run(_ options: Options) async throws {
    let start = ContinuousClock.now
    let referenceURL = URL(fileURLWithPath: options.reference)
    let size = try referenceURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
    guard size > 0 && size <= 16 * 1024 * 1024 else { throw ClientError("Reference exceeds file limit") }
    let reference = try Data(contentsOf: referenceURL)
    let body = try JSONSerialization.data(withJSONObject: [
        "model": options.model, "input": options.text, "ref_audio": reference.base64EncodedString(),
        "response_format": options.format, "stream": options.format == "pcm",
    ])
    let transport = try Transport(options: options, body: body)
    defer { transport.close() }
    var next = try await transport.read()
    let response = try transport.response(url: URL(string: options.endpoint)!)
    guard response.statusCode == 200 else {
        var errorBody = next
        while !next.isEmpty && errorBody.count < 16384 {
            next = try await transport.read()
            errorBody.append(next)
        }
        throw ClientError("HTTP \(response.statusCode): \(String(decoding: errorBody, as: UTF8.self))")
    }
    try validateHeaders(response, format: options.format)
    let player = try Playback(start: start)
    var completed = false
    defer {
        if !completed {
            FileHandle.standardError.write(Data("playback stopped: played_frames=\(player.playedFrames), queued_frames=\(player.queuedFrames)\n".utf8))
        }
        player.stop()
    }
    let outputURL = options.output.map { URL(fileURLWithPath: $0) }
    let partialURL = outputURL.map { $0.deletingLastPathComponent().appendingPathComponent(".speech-\(UUID().uuidString).partial") }
    if let outputURL, FileManager.default.fileExists(atPath: outputURL.path) {
        throw ClientError("Output already exists; choose a new file")
    }
    var file: FileHandle?
    if let partialURL {
        guard FileManager.default.createFile(atPath: partialURL.path, contents: nil) else { throw ClientError("Cannot create output") }
        file = try FileHandle(forWritingTo: partialURL)
    }
    defer {
        try? file?.close()
        if let partialURL { try? FileManager.default.removeItem(at: partialURL) }
    }
    var decoder = PCMDecoder()
    var wav: [UInt8] = []
    var received = 0
    // Intentionally odd reads prove transport reads need not align to s16 samples.
    while !next.isEmpty {
        try Task.checkCancellation()
        received += next.count
        guard received <= 26_460_044 else { throw ClientError("Audio response exceeds limit") }
        if options.format == "wav" {
            wav.append(contentsOf: next)
        } else {
            try file?.write(contentsOf: next)
            try await player.append(decoder.decode(Array(next)))
        }
        next = try await transport.read()
    }
    try await transport.finish()
    let receivedMS = elapsedMS(start)
    if options.format == "wav" {
        guard response.expectedContentLength == received else { throw ClientError("WAV Content-Length mismatch") }
        let pcm = try wavPCM(wav)
        guard !pcm.isEmpty else { throw ClientError("Empty audio") }
        try file?.write(contentsOf: Data(wav))
        for offset in stride(from: 0, to: pcm.count, by: 777) {
            try await player.append(decoder.decode(Array(pcm[offset..<min(offset + 777, pcm.count)])))
        }
    } else {
        guard received > 0 else { throw ClientError("Empty audio") }
    }
    try decoder.finish()
    try await player.finish()
    try Task.checkCancellation()
    try file?.close()
    file = nil
    if let partialURL, let outputURL { try FileManager.default.moveItem(at: partialURL, to: outputURL) }
    let result: [String: Any] = [
        "status": "played", "format": options.format, "received_bytes": received,
        "played_frames": player.playedFrames, "peak_queued_frames": player.peakQueuedFrames,
        "first_played_ms": player.firstPlayedMS ?? -1, "download_complete_ms": receivedMS,
        "playback_complete_ms": elapsedMS(start),
    ]
    print(String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), as: UTF8.self))
    completed = true
}

@main
struct SpeechClient {
    @MainActor static func main() async {
        do {
            let options = try Options(Array(CommandLine.arguments.dropFirst()))
            let task = Task { try await run(options) }
            signal(SIGINT, SIG_IGN)
            let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
            interrupt.setEventHandler { task.cancel() }
            interrupt.resume()
            let timer = Task {
                if let delay = options.cancelAfterMS {
                    try await Task.sleep(for: .milliseconds(delay))
                    task.cancel()
                }
            }
            defer { timer.cancel(); interrupt.cancel() }
            try await task.value
        } catch {
            FileHandle.standardError.write(Data("speech client failed: \(error)\n".utf8))
            exit(1)
        }
    }
}
