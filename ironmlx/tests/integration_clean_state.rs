//! 调用 verify_clean_state 的独立诊断测试，用于检查 GPU/内存清理情况。

mod common;

use common::clean_state::verify_clean_state;

#[test]
#[ignore]
fn integration_clean_state() {
    match verify_clean_state("sweep-inter-suite") {
        Ok(report) => println!("clean state OK: {report:#?}"),
        Err(e) => {
            println!("clean state DEGRADED: {e}");
            // Report diagnostics without failing; callers decide whether to block.
        }
    }
}
