//! B1-p2.5 G4: 调用 verify_clean_state 的独立测试，由 sweep_full.sh
//! 在 suite 间调用以检查 GPU/内存清理情况。

mod common;

use common::clean_state::verify_clean_state;

#[test]
#[ignore]
fn integration_clean_state() {
    match verify_clean_state("sweep-inter-suite") {
        Ok(report) => println!("clean state OK: {report:#?}"),
        Err(e) => {
            println!("clean state DEGRADED: {e}");
            // Don't fail the test — sweep_full.sh logs the output; human/automation
            // decides whether to block on this. Informational only.
        }
    }
}
