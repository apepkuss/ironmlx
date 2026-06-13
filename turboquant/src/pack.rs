//! Bit-packing utilities for 3-bit and 4-bit indices.
//!
//! turbo3: 32 x 3-bit indices -> 12 bytes (96 bits)
//! turbo4: 32 x 4-bit indices -> 16 bytes (128 bits)

/// Pack 32 3-bit indices (each 0..7) into 12 bytes.
///
/// Packing layout: sequential bits, little-endian byte order.
/// indices[0] occupies bits 0..2 of byte 0, indices[1] bits 3..5, etc.
pub fn pack_3bit(indices: &[u8; 32]) -> [u8; 12] {
    let mut packed = [0u8; 12];
    let mut bit_pos = 0usize;
    for &idx in indices {
        debug_assert!(idx < 8, "3-bit index out of range: {idx}");
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        packed[byte_idx] |= (idx & 0x07) << bit_offset;
        // Handle overflow into next byte
        if bit_offset > 5 {
            packed[byte_idx + 1] |= (idx & 0x07) >> (8 - bit_offset);
        }
        bit_pos += 3;
    }
    packed
}

/// Unpack 12 bytes into 32 3-bit indices.
pub fn unpack_3bit(packed: &[u8; 12]) -> [u8; 32] {
    let mut indices = [0u8; 32];
    let mut bit_pos = 0usize;
    for idx in &mut indices {
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mut val = (packed[byte_idx] >> bit_offset) & 0x07;
        // Handle overflow from next byte
        if bit_offset > 5 && byte_idx + 1 < 12 {
            val |= (packed[byte_idx + 1] << (8 - bit_offset)) & 0x07;
        }
        *idx = val;
        bit_pos += 3;
    }
    indices
}

/// Pack 32 4-bit indices (each 0..15) into 16 bytes.
pub fn pack_4bit(indices: &[u8; 32]) -> [u8; 16] {
    let mut packed = [0u8; 16];
    for i in 0..16 {
        debug_assert!(indices[i * 2] < 16 && indices[i * 2 + 1] < 16);
        packed[i] = (indices[i * 2] & 0x0F) | ((indices[i * 2 + 1] & 0x0F) << 4);
    }
    packed
}

/// Unpack 16 bytes into 32 4-bit indices.
pub fn unpack_4bit(packed: &[u8; 16]) -> [u8; 32] {
    let mut indices = [0u8; 32];
    for i in 0..16 {
        indices[i * 2] = packed[i] & 0x0F;
        indices[i * 2 + 1] = (packed[i] >> 4) & 0x0F;
    }
    indices
}

/// Bytes required to pack `n` indices at given bit-width (block_size=32).
pub const fn packed_bytes(bits: u8) -> usize {
    match bits {
        3 => 12, // 32 * 3 / 8
        4 => 16, // 32 * 4 / 8
        _ => panic!("unsupported bit-width"),
    }
}

/// Block size (number of values per quantization block).
pub const BLOCK_SIZE: usize = 32;

/// Total bytes per turbo block (scale + packed indices).
pub const fn block_bytes(bits: u8) -> usize {
    2 + packed_bytes(bits) // 2 bytes f16 scale + packed indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_3bit_roundtrip() {
        let indices: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0, 0, 3, 5, 7, 1, 4, 6, 2, 3, 3, 3, 3, 7,
            0, 5, 2,
        ];
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_3bit_all_zeros() {
        let indices = [0u8; 32];
        let packed = pack_3bit(&indices);
        assert_eq!(packed, [0u8; 12]);
        assert_eq!(unpack_3bit(&packed), indices);
    }

    #[test]
    fn pack_unpack_3bit_all_sevens() {
        let indices = [7u8; 32];
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_roundtrip() {
        let indices: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7,
            6, 5, 4, 3, 2, 1, 0,
        ];
        let packed = pack_4bit(&indices);
        let unpacked = unpack_4bit(&packed);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn packed_sizes_correct() {
        assert_eq!(packed_bytes(3), 12);
        assert_eq!(packed_bytes(4), 16);
        assert_eq!(block_bytes(3), 14); // 2 + 12
        assert_eq!(block_bytes(4), 18); // 2 + 16
    }
}
