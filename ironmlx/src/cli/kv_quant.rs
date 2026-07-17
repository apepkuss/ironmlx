use clap::ValueEnum;

use crate::core::cache::TurboQuantKVBits;
use crate::core::scheduler_autotune::SchedulerKvQuantization;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum KvQuantArg {
    None,
    Turbo3,
    Turbo4,
    #[value(name = "k3v4")]
    K3V4,
}

impl KvQuantArg {
    pub(crate) fn cli_value(self) -> &'static str {
        match self {
            KvQuantArg::None => "none",
            KvQuantArg::Turbo3 => "turbo3",
            KvQuantArg::Turbo4 => "turbo4",
            KvQuantArg::K3V4 => "k3v4",
        }
    }

    pub(crate) fn profile_context(self) -> SchedulerKvQuantization {
        match self {
            KvQuantArg::None => SchedulerKvQuantization::None,
            KvQuantArg::Turbo3 => SchedulerKvQuantization::Turbo3,
            KvQuantArg::Turbo4 => SchedulerKvQuantization::Turbo4,
            KvQuantArg::K3V4 => SchedulerKvQuantization::K3V4,
        }
    }

    pub(crate) fn turboquant_bits(self) -> Option<TurboQuantKVBits> {
        match self {
            KvQuantArg::None => None,
            KvQuantArg::Turbo3 => Some(TurboQuantKVBits::K3V3),
            KvQuantArg::Turbo4 => Some(TurboQuantKVBits::K4V4),
            KvQuantArg::K3V4 => Some(TurboQuantKVBits::K3V4),
        }
    }
}
