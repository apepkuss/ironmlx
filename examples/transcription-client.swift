// Build: swiftc -parse-as-library examples/transcription-client.swift -o /tmp/ironmlx-transcription-client
// Input: raw PCM16 little-endian, mono, 24 kHz. This is an upstream API client, not App UI.
import Foundation
import Darwin

private struct TranscriptionClientError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

private func send(_ event: [String: Any], to socket: URLSessionWebSocketTask) async throws {
    let data = try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])
    try await socket.send(.string(String(decoding: data, as: UTF8.self)))
}

private func receive(from socket: URLSessionWebSocketTask) async throws -> [String: Any] {
    let data: Data
    switch try await socket.receive() {
    case .string(let text): data = Data(text.utf8)
    case .data(let bytes): data = bytes
    @unknown default: throw TranscriptionClientError("Unsupported WebSocket message")
    }
    guard let event = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw TranscriptionClientError("Expected a JSON event")
    }
    if event["type"] as? String == "error"
        || event["type"] as? String == "conversation.item.input_audio_transcription.failed" {
        throw TranscriptionClientError(String(decoding: data, as: UTF8.self))
    }
    return event
}

@main
private struct TranscriptionClient {
    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(EXIT_FAILURE)
        }
    }

    private static func run() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count.isMultiple(of: 2) else {
            throw TranscriptionClientError("Use --audio FILE.pcm [--model ID] [--url ws://HOST/v1/realtime] [--language zh] [--diarization true|false]")
        }
        var options: [String: String] = [:]
        for index in stride(from: 0, to: arguments.count, by: 2) {
            let name = arguments[index]
            guard ["--audio", "--model", "--url", "--language", "--diarization"].contains(name), options[name] == nil else {
                throw TranscriptionClientError("Unknown or repeated option: \(name)")
            }
            options[name] = arguments[index + 1]
        }
        guard let path = options["--audio"] else {
            throw TranscriptionClientError("--audio must name a raw PCM16 mono 24 kHz file")
        }
        let audio = try Data(contentsOf: URL(fileURLWithPath: path))
        guard audio.count >= 4800, audio.count.isMultiple(of: 2), audio.count <= 7200 * 48000,
              audio.prefix(4) != Data("RIFF".utf8) else {
            throw TranscriptionClientError("Expected 100 ms–2 hours of raw PCM16 mono 24 kHz, without a WAV header")
        }
        guard options["--diarization"] == nil || ["true", "false"].contains(options["--diarization"]!) else {
            throw TranscriptionClientError("--diarization must be true or false")
        }
        let diarization = options["--diarization"] == "true"
        let model = options["--model"] ?? "mlx-community/Qwen3-ASR-0.6B-8bit"
        guard var url = URLComponents(string: options["--url"] ?? "ws://127.0.0.1:9068/v1/realtime"),
              ["ws", "wss"].contains(url.scheme ?? "") else {
            throw TranscriptionClientError("Expected a ws:// or wss:// URL")
        }
        url.queryItems = [URLQueryItem(name: "intent", value: "transcription")]
        guard let endpoint = url.url else { throw TranscriptionClientError("Invalid URL") }
        var request = URLRequest(url: endpoint)
        if let key = ProcessInfo.processInfo.environment["IRONMLX_API_KEY"], !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        let socket = URLSession.shared.webSocketTask(with: request)
        socket.resume()
        defer { socket.cancel(with: .normalClosure, reason: nil) }
        guard try await receive(from: socket)["type"] as? String == "session.created" else {
            throw TranscriptionClientError("Expected session.created")
        }
        var transcription = ["model": model]
        if let language = options["--language"] { transcription["language"] = language }
        var input: [String: Any] = [
            "format": ["type": "audio/pcm", "rate": 24000],
            "transcription": transcription, "turn_detection": NSNull()
        ]
        if diarization { input["diarization"] = [String: Any]() }
        try await send(["type": "session.update", "session": ["type": "transcription", "audio": ["input": input]]], to: socket)
        guard try await receive(from: socket)["type"] as? String == "session.updated" else {
            throw TranscriptionClientError("Expected session.updated")
        }
        // Receive while sending, so interim results are visible before the recording ends.
        let results = Task {
            while true {
                let event = try await receive(from: socket)
                let type = event["type"] as? String ?? ""
                let data = try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])
                let finalType = diarization ? "conversation.item.input_audio_transcription.diarized" : "conversation.item.input_audio_transcription.completed"
                if type == finalType {
                    FileHandle.standardOutput.write(data + Data([10]))
                    // IronMLX marks the final item of an explicitly committed turn.
                    if event["is_final"] as? Bool == true { return }
                } else if type == "conversation.item.input_audio_transcription.partial"
                    || type == "input_audio_buffer.diarization.delta"
                    || type == "input_audio_buffer.diarization.started" {
                    FileHandle.standardError.write(data + Data([10]))
                }
            }
        }
        defer { results.cancel() }
        for start in stride(from: 0, to: audio.count, by: 9600) {
            let end = min(start + 9600, audio.count)
            try await send(["type": "input_audio_buffer.append", "audio": audio[start..<end].base64EncodedString()], to: socket)
            try await Task.sleep(nanoseconds: UInt64(end - start) * 1_000_000_000 / 48000)
        }
        try await send(["type": "input_audio_buffer.commit"], to: socket)
        try await results.value
    }
}
