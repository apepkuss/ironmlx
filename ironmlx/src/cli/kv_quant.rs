use clap::ValueEnum;

use crate::core::cache::TurboQuantKVBits;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum KvQuantArg {
    None,
    Turbo3,
    Turbo4,
    #[value(name = "k3v4")]
    K3V4,
}

impl KvQuantArg {
    pub(crate) fn turboquant_bits(self) -> Option<TurboQuantKVBits> {
        match self {
            KvQuantArg::None => None,
            KvQuantArg::Turbo3 => Some(TurboQuantKVBits::K3V3),
            KvQuantArg::Turbo4 => Some(TurboQuantKVBits::K4V4),
            KvQuantArg::K3V4 => Some(TurboQuantKVBits::K3V4),
        }
    }
}
