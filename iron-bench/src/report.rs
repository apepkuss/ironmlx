//! Stats reduction + Markdown / CSV / JSON formatters. Stub bodies in T3;
//! filled in T4.

use crate::runner::CellResult;

pub fn render_markdown(
    _cells: &[CellResult],
    _targets: &[(String, String)],
    _warmup: usize,
) -> String {
    String::new()
}

pub fn render_csv(_cells: &[CellResult]) -> String {
    String::new()
}

pub fn render_json(_cells: &[CellResult], _targets: &[(String, String)], _warmup: usize) -> String {
    String::new()
}
