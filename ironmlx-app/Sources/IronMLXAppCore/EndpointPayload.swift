import Foundation

public struct EndpointPayload: Codable, Equatable {
    public struct NetworkInterface: Codable, Equatable {
        public var name: String
        public var ip: String
        public var isThunderbolt: Bool

        public init(name: String, ip: String, isThunderbolt: Bool = false) {
            self.name = name
            self.ip = ip
            self.isThunderbolt = isThunderbolt
        }

        enum CodingKeys: String, CodingKey {
            case name
            case ip
            case isThunderbolt = "is_thunderbolt"
        }
    }

    public var host: String
    public var loopback: String
    public var port: UInt16
    public var interfaces: [NetworkInterface]

    public init(host: String, port: UInt16, interfaces: [NetworkInterface] = []) {
        self.host = host
        self.loopback = "127.0.0.1"
        self.port = port
        self.interfaces = interfaces
    }
}
