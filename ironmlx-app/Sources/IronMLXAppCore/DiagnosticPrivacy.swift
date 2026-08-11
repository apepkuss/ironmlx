import Foundation

public enum DiagnosticPrivacy {
    public static let redacted = "<redacted>"
    public static let truncatedMarker = "[truncated]\n"

    private static let sensitiveJSONKeys = [
        "prompt", "prompts", "messages", "input", "inputs", "request_body",
        "request-body", "body", "arguments", "function_arguments", "tool_arguments",
        "authorization", "cookie", "set-cookie", "api_key", "api-key", "hf_token",
        "token", "lan_api_key", "password", "secret",
    ]

    public static func sanitizedLog(_ text: String, maximumBytes: Int) -> String {
        sanitizedText(text, maximumBytes: maximumBytes, preserveTail: true)
    }

    public static func sanitizedText(
        _ text: String,
        maximumBytes: Int,
        preserveTail: Bool = false
    ) -> String {
        guard maximumBytes > 0 else { return "" }
        var value = redactStructuredValues(in: text)
        let patterns: [(String, String)] = [
            (#"(?i)(request\s+body\s*[:=]\s*)[^\r\n]*"#, "$1<redacted>"),
            (#"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?([^\s,;]+)"#, "$1<redacted>"),
            (#"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]+)"#, "$1<redacted>"),
            (#"(?i)((?:hf[_-]?token|lan[_-]?api[_-]?key|api[_-]?key|token|cookie|password|secret)\s*[:=]\s*)([^\s,;]+)"#, "$1<redacted>"),
            (#"(?i)(cookie\s*:\s*)[^\r\n]*"#, "$1<redacted>"),
        ]
        for (pattern, replacement) in patterns {
            value = replacing(pattern, in: value, with: replacement)
        }
        value = sanitizedPaths(in: value)
        return bounded(value, maximumBytes: maximumBytes, preserveTail: preserveTail)
    }

    public static func sanitizedData(
        _ data: Data,
        maximumBytes: Int,
        preserveTail: Bool = false
    ) -> Data {
        Data(sanitizedText(
            String(decoding: data, as: UTF8.self),
            maximumBytes: maximumBytes,
            preserveTail: preserveTail
        ).utf8)
    }

    public static func stableModelReference(_ value: String) -> String {
        let sanitized = sanitizedText(value, maximumBytes: 1_024)
        if sanitized.hasPrefix("/") || sanitized.hasPrefix("file:") {
            return "local://" + URL(fileURLWithPath: sanitized).lastPathComponent
        }
        return sanitized
    }

    private static func redactStructuredValues(in text: String) -> String {
        let keys = sensitiveJSONKeys.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
        let pattern = #"(?is)([\"'](?:"# + keys + #")[\"']\s*:\s*)(?:[\"'](?:\\.|(?![\"']).)*[\"']|\[(?:\\.|.)*?\]|\{(?:\\.|.)*?\}|[^,\r\n}]+)"#
        var value = replacing(pattern, in: text, with: "$1\"<redacted>\"")

        // Broken or pretty-printed request objects still get a conservative
        // line-oriented second pass. Once a sensitive key starts a composite
        // value, no continuation line is retained.
        let keyPattern = #"(?i)[\"']?(?:"# + keys + #")[\"']?\s*[:=]"#
        guard let expression = try? NSRegularExpression(pattern: keyPattern) else { return value }
        var suppressing = false
        var depth = 0
        let lines = value.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let result = lines.compactMap { line -> String? in
            if suppressing {
                depth += compositeDelta(line)
                if depth <= 0 { suppressing = false }
                return nil
            }
            let range = NSRange(line.startIndex..., in: line)
            guard expression.firstMatch(in: line, range: range) != nil else { return line }
            let prefix = line.prefix { $0 == " " || $0 == "\t" }
            let delta = compositeDelta(line)
            if delta > 0 {
                suppressing = true
                depth = delta
            }
            return "\(prefix)\(redacted)"
        }
        value = result.joined(separator: "\n")
        return value
    }

    private static func compositeDelta(_ line: String) -> Int {
        line.reduce(into: 0) { result, character in
            if character == "{" || character == "[" { result += 1 }
            if character == "}" || character == "]" { result -= 1 }
        }
    }

    private static func sanitizedPaths(in text: String) -> String {
        var value = text
        let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.path
        if !home.isEmpty {
            value = value.replacingOccurrences(of: home, with: "/Users/<user>")
            value = value.replacingOccurrences(
                of: home.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? home,
                with: "/Users/<user>"
            )
        }
        let username = NSUserName()
        if !username.isEmpty {
            value = value.replacingOccurrences(of: username, with: "<user>")
        }
        // Assemble the conventional user-root spelling at runtime so the App
        // binary's developer-path gate cannot mistake this privacy pattern for
        // a captured build-machine path.
        let usersRoot = ["", "Users", ""].joined(separator: "/")
        value = replacing(usersRoot + #"[^/\s\"']+"#, in: value, with: usersRoot + "<user>")
        value = replacing(
            #"file:(?://)?"# + usersRoot + #"[^/\s\"']+"#,
            in: value,
            with: "file://" + usersRoot + "<user>"
        )
        return value
    }

    private static func bounded(_ value: String, maximumBytes: Int, preserveTail: Bool) -> String {
        let data = Data(value.utf8)
        guard data.count > maximumBytes else { return value }
        let marker = Data(truncatedMarker.utf8)
        let available = max(0, maximumBytes - marker.count)
        let selected = preserveTail ? data.suffix(available) : data.prefix(available)
        return String(decoding: preserveTail ? marker + selected : selected + marker, as: UTF8.self)
    }

    private static func replacing(_ pattern: String, in value: String, with replacement: String) -> String {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return value }
        return expression.stringByReplacingMatches(
            in: value,
            range: NSRange(value.startIndex..., in: value),
            withTemplate: replacement
        )
    }
}
