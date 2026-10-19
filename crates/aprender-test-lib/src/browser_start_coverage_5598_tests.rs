//! Coverage for the CDP `Page::start_coverage_with_config` (#5598).
//!
//! Only the page-without-a-browser-handle path is reachable without chromium:
//! `Page::new` leaves `inner` as `None`.

use crate::{CoverageConfig, Page};

#[tokio::test]
async fn without_a_browser_handle_coverage_stays_disabled() {
    let mut page = Page::new(800, 600);
    page.start_coverage_with_config(CoverageConfig::default())
        .await
        .expect("no-op is Ok");
    assert!(
        !page.is_coverage_enabled(),
        "no CDP page, so nothing was instrumented"
    );
}

#[tokio::test]
async fn start_coverage_delegates_and_matches_the_with_config_path() {
    let mut a = Page::new(1, 1);
    let mut b = Page::new(1, 1);
    a.start_coverage().await.expect("ok");
    b.start_coverage_with_config(CoverageConfig::default())
        .await
        .expect("ok");
    assert_eq!(a.is_coverage_enabled(), b.is_coverage_enabled());
}

#[tokio::test]
async fn config_flags_do_not_change_the_no_handle_outcome() {
    let mut page = Page::new(10, 10);
    let cfg = CoverageConfig {
        call_count: false,
        detailed: true,
        allow_triggered_updates: true,
    };
    assert!(page.start_coverage_with_config(cfg).await.is_ok());
    assert!(!page.is_coverage_enabled());
}
