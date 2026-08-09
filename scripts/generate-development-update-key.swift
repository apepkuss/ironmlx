#!/usr/bin/env swift

import CryptoKit
import Foundation

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(
        Data("usage: generate-development-update-key.swift <private-key-output>\n".utf8)
    )
    exit(2)
}

let outputURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard outputURL.path.hasPrefix("/") else {
    FileHandle.standardError.write(Data("error: private key output must be an absolute path\n".utf8))
    exit(2)
}
guard !FileManager.default.fileExists(atPath: outputURL.path) else {
    FileHandle.standardError.write(Data("error: refusing to overwrite an existing private key\n".utf8))
    exit(2)
}

let privateKey = Curve25519.Signing.PrivateKey()
let privateKeyText = privateKey.rawRepresentation.base64EncodedString() + "\n"
try Data(privateKeyText.utf8).write(to: outputURL, options: .withoutOverwriting)
try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: outputURL.path)
print(privateKey.publicKey.rawRepresentation.base64EncodedString())
