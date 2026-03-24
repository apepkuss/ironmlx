//! Internationalization — shared language state and translation.

use std::sync::{Mutex, OnceLock};

static NAV_LANGUAGE: OnceLock<Mutex<&'static str>> = OnceLock::new();

pub fn nav_language() -> &'static Mutex<&'static str> {
    NAV_LANGUAGE.get_or_init(|| Mutex::new("en"))
}

/// Get translated string for current language (menu items).
pub fn t(key: &str) -> &str {
    let lang = *nav_language().lock().unwrap();
    match (lang, key) {
        // Chinese Simplified
        ("zh", "menu_dashboard") => "\u{4eea}\u{8868}\u{76d8}",
        ("zh", "menu_chat") => "\u{4e0e} ironmlx \u{5bf9}\u{8bdd}",
        ("zh", "menu_stop") => "\u{505c}\u{6b62}\u{670d}\u{52a1}",
        ("zh", "menu_start") => "\u{542f}\u{52a8}\u{670d}\u{52a1}",
        ("zh", "menu_restart") => "\u{91cd}\u{542f}\u{670d}\u{52a1}",
        ("zh", "menu_preferences") => "\u{504f}\u{597d}\u{8bbe}\u{7f6e}...",
        ("zh", "menu_updates") => "\u{68c0}\u{67e5}\u{66f4}\u{65b0}...",
        ("zh", "menu_quit") => "\u{9000}\u{51fa}",
        ("zh", "menu_status_running") => "\u{670d}\u{52a1}\u{5668}\u{8fd0}\u{884c}\u{4e2d}",
        ("zh", "menu_status_stopped") => "\u{670d}\u{52a1}\u{5668}\u{5df2}\u{505c}\u{6b62}",
        // Chinese Traditional
        ("zh-Hant", "menu_dashboard") => "\u{5100}\u{8868}\u{677f}",
        ("zh-Hant", "menu_chat") => "\u{8207} ironmlx \u{5c0d}\u{8a71}",
        ("zh-Hant", "menu_stop") => "\u{505c}\u{6b62}\u{670d}\u{52d9}",
        ("zh-Hant", "menu_start") => "\u{555f}\u{52d5}\u{670d}\u{52d9}",
        ("zh-Hant", "menu_restart") => "\u{91cd}\u{555f}\u{670d}\u{52d9}",
        ("zh-Hant", "menu_preferences") => "\u{504f}\u{597d}\u{8a2d}\u{5b9a}...",
        ("zh-Hant", "menu_updates") => "\u{6aa2}\u{67e5}\u{66f4}\u{65b0}...",
        ("zh-Hant", "menu_quit") => "\u{9000}\u{51fa}",
        ("zh-Hant", "menu_status_running") => "\u{4f3a}\u{670d}\u{5668}\u{57f7}\u{884c}\u{4e2d}",
        ("zh-Hant", "menu_status_stopped") => "\u{4f3a}\u{670d}\u{5668}\u{5df2}\u{505c}\u{6b62}",
        // Japanese
        ("ja", "menu_dashboard") => "\u{30c0}\u{30c3}\u{30b7}\u{30e5}\u{30dc}\u{30fc}\u{30c9}",
        ("ja", "menu_chat") => "ironmlx \u{3068}\u{30c1}\u{30e3}\u{30c3}\u{30c8}",
        ("ja", "menu_stop") => "\u{30b5}\u{30fc}\u{30d0}\u{30fc}\u{505c}\u{6b62}",
        ("ja", "menu_start") => "\u{30b5}\u{30fc}\u{30d0}\u{30fc}\u{958b}\u{59cb}",
        ("ja", "menu_restart") => "\u{30b5}\u{30fc}\u{30d0}\u{30fc}\u{518d}\u{8d77}\u{52d5}",
        ("ja", "menu_preferences") => "\u{74b0}\u{5883}\u{8a2d}\u{5b9a}...",
        ("ja", "menu_updates") => {
            "\u{30a2}\u{30c3}\u{30d7}\u{30c7}\u{30fc}\u{30c8}\u{78ba}\u{8a8d}..."
        }
        ("ja", "menu_quit") => "\u{7d42}\u{4e86}",
        ("ja", "menu_status_running") => {
            "\u{30b5}\u{30fc}\u{30d0}\u{30fc}: \u{5b9f}\u{884c}\u{4e2d}"
        }
        ("ja", "menu_status_stopped") => {
            "\u{30b5}\u{30fc}\u{30d0}\u{30fc}: \u{505c}\u{6b62}\u{4e2d}"
        }
        // Korean
        ("ko", "menu_dashboard") => "\u{b300}\u{c2dc}\u{bcf4}\u{b4dc}",
        ("ko", "menu_chat") => "ironmlx\u{c640} \u{cc44}\u{d305}",
        ("ko", "menu_stop") => "\u{c11c}\u{bc84} \u{c815}\u{c9c0}",
        ("ko", "menu_start") => "\u{c11c}\u{bc84} \u{c2dc}\u{c791}",
        ("ko", "menu_restart") => "\u{c11c}\u{bc84} \u{c7ac}\u{c2dc}\u{c791}",
        ("ko", "menu_preferences") => "\u{d658}\u{acbd}\u{c124}\u{c815}...",
        ("ko", "menu_updates") => "\u{c5c5}\u{b370}\u{c774}\u{d2b8} \u{d655}\u{c778}...",
        ("ko", "menu_quit") => "\u{c885}\u{b8cc}",
        ("ko", "menu_status_running") => "\u{c11c}\u{bc84}: \u{c2e4}\u{d589} \u{c911}",
        ("ko", "menu_status_stopped") => "\u{c11c}\u{bc84}: \u{c815}\u{c9c0}\u{b428}",
        // English defaults
        (_, "menu_dashboard") => "Dashboard",
        (_, "menu_chat") => "Chat with ironmlx",
        (_, "menu_stop") => "Stop Server",
        (_, "menu_start") => "Start Server",
        (_, "menu_restart") => "Restart Server",
        (_, "menu_preferences") => "Preferences...",
        (_, "menu_updates") => "Check for Updates...",
        (_, "menu_quit") => "Quit",
        (_, "menu_status_running") => "Server: Running",
        (_, "menu_status_stopped") => "Server: Stopped",
        _ => key,
    }
}
