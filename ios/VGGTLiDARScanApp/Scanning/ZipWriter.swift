import Foundation

enum ZipWriter {
    static func zipDirectory(_ directoryURL: URL, to outputURL: URL) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }
        fileManager.createFile(atPath: outputURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: outputURL)
        defer { try? handle.close() }

        var entries: [CentralDirectoryEntry] = []
        let files = try recursiveFiles(in: directoryURL)
        for fileURL in files {
            let relativePath = fileURL.path.replacingOccurrences(of: directoryURL.path + "/", with: "")
            let data = try Data(contentsOf: fileURL)
            let offset = try handle.offset()
            let crc = CRC32.checksum(data)
            try writeLocalFileHeader(to: handle, path: relativePath, crc: crc, size: UInt32(data.count))
            handle.write(data)
            entries.append(CentralDirectoryEntry(path: relativePath, crc: crc, size: UInt32(data.count), localHeaderOffset: UInt32(offset)))
        }

        let centralDirectoryOffset = try handle.offset()
        for entry in entries {
            try writeCentralDirectoryHeader(to: handle, entry: entry)
        }
        let centralDirectorySize = try handle.offset() - centralDirectoryOffset
        try writeEndOfCentralDirectory(
            to: handle,
            entryCount: UInt16(entries.count),
            centralDirectorySize: UInt32(centralDirectorySize),
            centralDirectoryOffset: UInt32(centralDirectoryOffset)
        )
    }

    private static func recursiveFiles(in directoryURL: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return try enumerator.compactMap { item in
            guard let url = item as? URL else { return nil }
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            return values.isRegularFile == true ? url : nil
        }
        .sorted { $0.path < $1.path }
    }

    private static func writeLocalFileHeader(to handle: FileHandle, path: String, crc: UInt32, size: UInt32) throws {
        let pathData = Data(path.utf8)
        var data = Data()
        data.appendUInt32LE(0x04034b50)
        data.appendUInt16LE(20)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt32LE(crc)
        data.appendUInt32LE(size)
        data.appendUInt32LE(size)
        data.appendUInt16LE(UInt16(pathData.count))
        data.appendUInt16LE(0)
        data.append(pathData)
        handle.write(data)
    }

    private static func writeCentralDirectoryHeader(to handle: FileHandle, entry: CentralDirectoryEntry) throws {
        let pathData = Data(entry.path.utf8)
        var data = Data()
        data.appendUInt32LE(0x02014b50)
        data.appendUInt16LE(20)
        data.appendUInt16LE(20)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt32LE(entry.crc)
        data.appendUInt32LE(entry.size)
        data.appendUInt32LE(entry.size)
        data.appendUInt16LE(UInt16(pathData.count))
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt32LE(0)
        data.appendUInt32LE(entry.localHeaderOffset)
        data.append(pathData)
        handle.write(data)
    }

    private static func writeEndOfCentralDirectory(
        to handle: FileHandle,
        entryCount: UInt16,
        centralDirectorySize: UInt32,
        centralDirectoryOffset: UInt32
    ) throws {
        var data = Data()
        data.appendUInt32LE(0x06054b50)
        data.appendUInt16LE(0)
        data.appendUInt16LE(0)
        data.appendUInt16LE(entryCount)
        data.appendUInt16LE(entryCount)
        data.appendUInt32LE(centralDirectorySize)
        data.appendUInt32LE(centralDirectoryOffset)
        data.appendUInt16LE(0)
        handle.write(data)
    }
}

private struct CentralDirectoryEntry {
    let path: String
    let crc: UInt32
    let size: UInt32
    let localHeaderOffset: UInt32
}

private enum CRC32 {
    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xffffffff
        for byte in data {
            let index = Int((crc ^ UInt32(byte)) & 0xff)
            crc = (crc >> 8) ^ table[index]
        }
        return crc ^ 0xffffffff
    }

    private static let table: [UInt32] = (0..<256).map { index in
        var value = UInt32(index)
        for _ in 0..<8 {
            if value & 1 == 1 {
                value = 0xedb88320 ^ (value >> 1)
            } else {
                value >>= 1
            }
        }
        return value
    }
}

private extension Data {
    mutating func appendUInt16LE(_ value: UInt16) {
        append(UInt8(value & 0xff))
        append(UInt8((value >> 8) & 0xff))
    }

    mutating func appendUInt32LE(_ value: UInt32) {
        append(UInt8(value & 0xff))
        append(UInt8((value >> 8) & 0xff))
        append(UInt8((value >> 16) & 0xff))
        append(UInt8((value >> 24) & 0xff))
    }
}
