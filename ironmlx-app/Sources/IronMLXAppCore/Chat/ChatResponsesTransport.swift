import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Sends one streaming `/v1/responses` request and yields the raw SSE bytes.
protocol ChatResponsesTransport: Sendable {
    /// Throws `ChatTransportError.http` when the server answers with a non-2xx
    /// status (it does so before streaming, e.g. when the request does not fit
    /// the context). Cancelling the consuming task closes the connection.
    func stream(body: Data) async throws -> AsyncThrowingStream<Data, Error>
}

enum ChatTransportError: Error, Equatable {
    case http(status: Int, body: Data)
}

/// Talks to the backend's loopback listener, the same host and port
/// `BackendAPIClient` uses; that listener needs no API key.
struct URLSessionResponsesTransport: ChatResponsesTransport {
    private let endpoint: @Sendable () -> URL?

    init(endpoint: @escaping @Sendable () -> URL?) {
        self.endpoint = endpoint
    }

    /// Reads host and port at each request, so a changed port is followed.
    init(configStore: AppConfigStore) {
        endpoint = {
            let config = configStore.load()
            let client = BackendAPIClient(host: config.host, port: config.port)
            return URL(string: "http://\(client.host):\(client.port)/v1/responses")
        }
    }

    /// Prefill of a long history can keep the connection quiet for a while
    /// before the first token, so the idle timeout is generous.
    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 600
        configuration.timeoutIntervalForResource = 6 * 60 * 60
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: configuration)
    }()

    func stream(body: Data) async throws -> AsyncThrowingStream<Data, Error> {
        guard let url = endpoint() else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = body
        let (bytes, response) = try await Self.session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard (200..<300).contains(http.statusCode) else {
            var data = Data()
            for try await byte in bytes { data.append(byte) }
            throw ChatTransportError.http(status: http.statusCode, body: data)
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // One chunk per line keeps each SSE frame whole for the parser.
                    var line = Data()
                    for try await byte in bytes {
                        line.append(byte)
                        if byte == 0x0A {
                            continuation.yield(line)
                            line = Data()
                        }
                    }
                    if !line.isEmpty { continuation.yield(line) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Turns a dropped image into a JPEG no larger than `maxPixelSize` on its long
/// side, with orientation applied, so every attachment is one known format of
/// bounded size.
enum ChatImageNormalizer {
    static let maxPixelSize = 2048
    static let jpegQuality = 0.85

    static func jpegData(contentsOf url: URL) -> Data? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output, UTType.jpeg.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(
            destination, image, [kCGImageDestinationLossyCompressionQuality: jpegQuality] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else { return nil }
        return output as Data
    }
}
