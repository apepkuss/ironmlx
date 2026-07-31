import Foundation
#if canImport(Darwin)
import Darwin
#endif

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
    public var networkMode: String
    public var authentication: String
    public var endpointAddress: String

    public init(
        host: String,
        port: UInt16,
        interfaces: [NetworkInterface]? = nil,
        networkMode: String = "local",
        lanHost: String? = nil,
        authentication: String = "none"
    ) {
        self.host = host
        self.loopback = "127.0.0.1"
        self.port = port
        self.networkMode = networkMode
        self.authentication = authentication
        if networkMode == "lan", let lanHost {
            self.interfaces = interfaces ?? Self.localNetworkInterfaces().filter { $0.ip == lanHost }
            self.endpointAddress = "https://\(lanHost):\(port)/v1"
        } else {
            self.interfaces = []
            self.endpointAddress = "http://127.0.0.1:\(port)/v1"
        }
    }

    public static func shouldPublishNetworkInterfaces(for host: String) -> Bool {
        isSafeLANAddress(host)
    }

    public static func isSafeLANAddress(_ host: String) -> Bool {
        localNetworkInterfaces().contains { $0.ip == host }
    }

    enum CodingKeys: String, CodingKey {
        case host
        case loopback
        case port
        case interfaces
        case networkMode = "network_mode"
        case authentication
        case endpointAddress = "endpoint_address"
    }

    public static func localNetworkInterfaces() -> [NetworkInterface] {
        #if canImport(Darwin)
        var addressList: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&addressList) == 0, let firstAddress = addressList else {
            return []
        }
        defer { freeifaddrs(addressList) }

        var interfaces: [NetworkInterface] = []
        var seen: Set<String> = []
        var pointer: UnsafeMutablePointer<ifaddrs>? = firstAddress
        while let current = pointer {
            defer { pointer = current.pointee.ifa_next }
            guard let address = current.pointee.ifa_addr,
                  address.pointee.sa_family == UInt8(AF_INET) else {
                continue
            }

            let flags = Int32(current.pointee.ifa_flags)
            guard flags & IFF_UP != 0,
                  flags & IFF_LOOPBACK == 0 else {
                continue
            }

            var hostBuffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(
                address,
                socklen_t(address.pointee.sa_len),
                &hostBuffer,
                socklen_t(hostBuffer.count),
                nil,
                0,
                NI_NUMERICHOST
            )
            guard result == 0 else {
                continue
            }

            let name = String(cString: current.pointee.ifa_name)
            let ip = String(
                decoding: hostBuffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) },
                as: UTF8.self
            )
            guard !ip.isEmpty,
                  !ip.hasPrefix("127."),
                  !seen.contains(ip) else {
                continue
            }
            seen.insert(ip)
            interfaces.append(NetworkInterface(name: name, ip: ip, isThunderbolt: isThunderboltInterface(name)))
        }
        return interfaces.sorted { lhs, rhs in
            if lhs.name == rhs.name {
                return lhs.ip < rhs.ip
            }
            return lhs.name < rhs.name
        }
        #else
        return []
        #endif
    }

    private static func isThunderboltInterface(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return lowered.contains("thunderbolt") || lowered.hasPrefix("bridge")
    }
}
