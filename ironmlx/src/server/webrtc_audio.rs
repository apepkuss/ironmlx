//! Bounded RTP reordering and native Opus decoding to the shared 24 kHz PCM input.
use std::time::{Duration, Instant};

pub(super) struct Packet {
    pub sequence: u16,
    pub timestamp: u32,
    pub payload: Vec<u8>,
    pub received: Instant,
}
pub(super) struct AudioDecoder {
    decoder: opus::Decoder,
    packets: Vec<Packet>,
    next_sequence: Option<u16>,
    next_timestamp: Option<u32>,
}
impl AudioDecoder {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            decoder: opus::Decoder::new(24000, opus::Channels::Mono)?,
            packets: Vec::new(),
            next_sequence: None,
            next_timestamp: None,
        })
    }
    pub fn push(&mut self, packet: Packet) -> anyhow::Result<()> {
        anyhow::ensure!(
            !packet.payload.is_empty() && packet.payload.len() <= 8192,
            "Opus packet exceeds 8192 bytes"
        );
        let next = *self.next_sequence.get_or_insert(packet.sequence);
        if (packet.sequence.wrapping_sub(next) as i16) < 0 {
            if self.next_timestamp.is_some() {
                return Ok(());
            }
            self.next_sequence = Some(packet.sequence);
        }
        if self.packets.iter().any(|p| p.sequence == packet.sequence) {
            return Ok(());
        }
        anyhow::ensure!(
            self.packets.len() < 64,
            "WebRTC audio jitter buffer is full"
        );
        self.packets.push(packet);
        Ok(())
    }
    /// A short hold admits reordered UDP packets; loss/DTX gaps retain RTP time.
    pub fn drain(&mut self, now: Instant, force: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut output = Vec::new();
        while let Some(next) = self.next_sequence {
            let Some(oldest) = self.packets.iter().map(|p| p.received).min() else {
                break;
            };
            let expired =
                force || now.saturating_duration_since(oldest) >= Duration::from_millis(60);
            if self.next_timestamp.is_none() && !expired {
                break;
            }
            let index = self
                .packets
                .iter()
                .position(|p| p.sequence == next)
                .or_else(|| {
                    expired
                        .then(|| {
                            self.packets
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, p)| p.sequence.wrapping_sub(next))
                                .map(|(i, _)| i)
                        })
                        .flatten()
                });
            let Some(index) = index else {
                break;
            };
            let packet = self.packets.swap_remove(index);
            self.next_sequence = Some(packet.sequence.wrapping_add(1));
            if let Some(expected) = self.next_timestamp {
                let ticks = packet.timestamp.wrapping_sub(expected) as i32;
                // Late/duplicate frames cannot move the accepted-audio clock backwards.
                if ticks < 0 {
                    continue;
                }
                anyhow::ensure!(
                    ticks <= 60 * 48000,
                    "WebRTC RTP audio clock jumped by more than 60 seconds"
                );
                let mut gap = ticks as usize / 2;
                while gap > 0 {
                    let frames = gap.min(4 * 24000);
                    output.push(vec![0; frames * 2]);
                    gap -= frames;
                }
            }
            let mut pcm = [0i16; 2880]; // Opus supports up to 120 ms at 24 kHz.
            let frames = self.decoder.decode(&packet.payload, &mut pcm, false)?;
            self.next_timestamp = Some(packet.timestamp.wrapping_add(frames as u32 * 2));
            output.push(pcm[..frames].iter().flat_map(|v| v.to_le_bytes()).collect());
        }
        Ok(output)
    }
    pub fn clear(&mut self) {
        self.packets.clear();
        self.next_sequence = None;
        self.next_timestamp = None;
        let _ = self.decoder.reset_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packets() -> Vec<Packet> {
        let mut encoder =
            opus::Encoder::new(24000, opus::Channels::Mono, opus::Application::Voip).unwrap();
        let now = Instant::now();
        (0..4)
            .map(|i| {
                let pcm = (0..480)
                    .map(|j| (((i * 480 + j) as f32 * 0.1).sin() * 10000.) as i16)
                    .collect::<Vec<_>>();
                let mut buffer = [0; 4000];
                let n = encoder.encode(&pcm, &mut buffer).unwrap();
                Packet {
                    sequence: 65534u16.wrapping_add(i as u16),
                    timestamp: (u32::MAX - 959).wrapping_add(i as u32 * 960),
                    payload: buffer[..n].to_vec(),
                    received: now,
                }
            })
            .collect()
    }
    #[test]
    fn reordered_duplicate_and_wrapped_rtp_preserves_decoded_audio() {
        let packets = packets();
        let mut normal = AudioDecoder::new().unwrap();
        let mut reordered = AudioDecoder::new().unwrap();
        for i in [0, 1, 2, 3] {
            let p = &packets[i];
            normal
                .push(Packet {
                    sequence: p.sequence,
                    timestamp: p.timestamp,
                    payload: p.payload.clone(),
                    received: p.received,
                })
                .unwrap();
        }
        for i in [1, 0, 3, 2, 2] {
            let p = &packets[i];
            reordered
                .push(Packet {
                    sequence: p.sequence,
                    timestamp: p.timestamp,
                    payload: p.payload.clone(),
                    received: p.received,
                })
                .unwrap();
        }
        assert_eq!(
            normal.drain(Instant::now(), true).unwrap(),
            reordered.drain(Instant::now(), true).unwrap()
        );
    }
    #[test]
    fn missing_packets_keep_silence_duration_and_clear_discards_buffered_audio() {
        let mut packets = packets();
        let last = packets.remove(3);
        let first = packets.remove(0);
        let mut decoder = AudioDecoder::new().unwrap();
        decoder.push(first).unwrap();
        decoder.push(last).unwrap();
        let output = decoder.drain(Instant::now(), true).unwrap();
        assert_eq!(output.iter().map(Vec::len).sum::<usize>(), 4 * 480 * 2);
        assert!(output[1].iter().all(|v| *v == 0));
        decoder.push(packets.remove(0)).unwrap();
        decoder.clear();
        assert!(decoder.drain(Instant::now(), true).unwrap().is_empty());
    }
}
