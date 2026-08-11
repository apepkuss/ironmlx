import Foundation
import Testing

@testable import IronMLXAppCore

@Test func diagnosticPrivacyRedactsStructuredMultilineRequestsCredentialsAndPaths() {
    let sentinels = [
        "PRIVATE_USER_PROMPT_SENTINEL", "HF_TOKEN_SENTINEL", "LAN_API_KEY_SENTINEL",
        "AUTHORIZATION_SENTINEL", "MULTILINE_MESSAGE_SENTINEL", "FUNCTION_ARGUMENT_SENTINEL",
    ]
    let source = """
    Authorization: Bearer AUTHORIZATION_SENTINEL
    hf_token=HF_TOKEN_SENTINEL
    lan_api_key=LAN_API_KEY_SENTINEL
    {
      "prompt": "PRIVATE_USER_PROMPT_SENTINEL",
      "messages": [
        {"role":"user","content":"MULTILINE_MESSAGE_SENTINEL"}
      ],
      "arguments": {"query":"FUNCTION_ARGUMENT_SENTINEL"}
    }
    path=\(NSHomeDirectory())/.ironmlx/models/private-model
    user=\(NSUserName())
    """

    let output = DiagnosticPrivacy.sanitizedText(source, maximumBytes: 64 * 1_024)

    for sentinel in sentinels {
        #expect(!output.contains(sentinel), "leaked \(sentinel)")
    }
    #expect(!output.contains(NSHomeDirectory()))
    #expect(!output.contains(NSUserName()))
    #expect(output.contains(DiagnosticPrivacy.redacted))
}

@Test func diagnosticPrivacyEnforcesByteCapacity() {
    let output = DiagnosticPrivacy.sanitizedLog(
        String(repeating: "backend-log-line-", count: 10_000),
        maximumBytes: 2_048
    )

    #expect(output.utf8.count <= 2_048)
    #expect(output.hasPrefix("[truncated]"))
}
