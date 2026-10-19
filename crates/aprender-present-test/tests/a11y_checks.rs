//! `A11yChecker::check_widget` (via `check_with_config`) and `AriaAttributes::to_html_attrs`
//! asserted rule by rule (#5704, #5708).
//!
//! Integration tests, not inline: `a11y.rs` already fails the pre-commit complexity hook on
//! `check_widget` itself (cognitive 29 > 25), so any edit to that file is refused.

use presentar_core::widget::{AccessibleRole, LayoutResult};
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Constraints, Event, Rect, Size,
    TypeId, Widget,
};
use presentar_test::{A11yChecker, A11yConfig, AriaAttributes, AriaChecked, AriaLive, Impact};
use std::any::Any;
use std::time::Duration;

struct W {
    interactive: bool,
    focusable: bool,
    name: Option<String>,
    role: AccessibleRole,
    bounds: Rect,
    children: Vec<Box<dyn Widget>>,
}

impl W {
    fn new() -> Self {
        Self {
            interactive: false,
            focusable: false,
            name: None,
            role: AccessibleRole::Generic,
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            children: Vec::new(),
        }
    }
    fn interactive(mut self) -> Self {
        self.interactive = true;
        self
    }
    fn focusable(mut self) -> Self {
        self.focusable = true;
        self
    }
    fn named(mut self, n: &str) -> Self {
        self.name = Some(n.to_string());
        self
    }
    fn role(mut self, r: AccessibleRole) -> Self {
        self.role = r;
        self
    }
    fn size(mut self, w: f32, h: f32) -> Self {
        self.bounds = Rect::new(0.0, 0.0, w, h);
        self
    }
    fn child(mut self, c: W) -> Self {
        self.children.push(Box::new(c));
        self
    }
}

impl Brick for W {
    fn brick_name(&self) -> &'static str {
        "W"
    }
    fn assertions(&self) -> &[BrickAssertion] {
        &[]
    }
    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }
    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![],
            failed: vec![],
            verification_time: Duration::from_micros(1),
        }
    }
    fn to_html(&self) -> String {
        String::new()
    }
    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for W {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
    fn measure(&self, c: Constraints) -> Size {
        c.constrain(Size::new(100.0, 50.0))
    }
    fn layout(&mut self, b: Rect) -> LayoutResult {
        LayoutResult { size: b.size() }
    }
    fn paint(&self, _: &mut dyn Canvas) {}
    fn event(&mut self, _: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }
    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }
    fn is_interactive(&self) -> bool {
        self.interactive
    }
    fn is_focusable(&self) -> bool {
        self.focusable
    }
    fn accessible_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn accessible_role(&self) -> AccessibleRole {
        self.role
    }
    fn bounds(&self) -> Rect {
        self.bounds
    }
}

fn rules(w: &W, cfg: &A11yConfig) -> Vec<String> {
    A11yChecker::check_with_config(w, cfg)
        .violations
        .into_iter()
        .map(|v| v.rule)
        .collect()
}

fn count(rs: &[String], rule: &str) -> usize {
    rs.iter().filter(|r| r.as_str() == rule).count()
}

#[test]
fn test_check_widget_interactive_unnamed_unfocusable_is_critical() {
    let w = W::new().interactive();
    let report = A11yChecker::check_with_config(&w, &A11yConfig::default());
    let by = |rule: &str| report.violations.iter().find(|v| v.rule == rule);
    let label = by("aria-label").expect("aria-label violation");
    assert_eq!(
        (label.wcag.as_str(), label.impact),
        ("4.1.2", Impact::Critical)
    );
    let kb = by("keyboard").expect("keyboard violation");
    assert_eq!((kb.wcag.as_str(), kb.impact), ("2.1.1", Impact::Critical));
    assert!(!report.is_passing());
}

#[test]
fn test_check_widget_named_focusable_interactive_passes() {
    let w = W::new()
        .interactive()
        .focusable()
        .named("Save")
        .size(48.0, 48.0);
    assert!(rules(&w, &A11yConfig::strict()).is_empty());
}

#[test]
fn test_check_widget_touch_target_boundary_and_config_toggle() {
    let ok = W::new()
        .interactive()
        .focusable()
        .named("a")
        .size(44.0, 44.0);
    assert_eq!(
        count(&rules(&ok, &A11yConfig::default()), "touch-target"),
        0
    );
    let narrow = W::new()
        .interactive()
        .focusable()
        .named("a")
        .size(43.9, 44.0);
    assert_eq!(
        count(&rules(&narrow, &A11yConfig::default()), "touch-target"),
        1
    );
    let short = W::new()
        .interactive()
        .focusable()
        .named("a")
        .size(44.0, 10.0);
    assert_eq!(
        count(&rules(&short, &A11yConfig::default()), "touch-target"),
        1
    );
    let off = A11yConfig {
        check_touch_targets: false,
        ..A11yConfig::default()
    };
    assert_eq!(count(&rules(&narrow, &off), "touch-target"), 0);
    // a non-interactive small widget is never a touch target
    let inert = W::new().size(5.0, 5.0);
    assert_eq!(
        count(&rules(&inert, &A11yConfig::default()), "touch-target"),
        0
    );
}

#[test]
fn test_check_widget_heading_order_skip_is_flagged_and_recorded() {
    let heading = |n: &str| W::new().role(AccessibleRole::Heading).named(n);
    let skipped = heading("h1").child(heading("h3"));
    let rs = rules(&skipped, &A11yConfig::default());
    assert_eq!(count(&rs, "heading-order"), 1, "{rs:?}");
    let in_order = heading("h1").child(heading("h2"));
    assert_eq!(
        count(&rules(&in_order, &A11yConfig::default()), "heading-order"),
        0
    );
    // level 1 -> 2 -> 4 skips exactly once
    let chain = heading("h1").child(heading("h2").child(heading("h4")));
    assert_eq!(
        count(&rules(&chain, &A11yConfig::default()), "heading-order"),
        1
    );
    let off = A11yConfig {
        check_heading_hierarchy: false,
        ..A11yConfig::default()
    };
    assert_eq!(count(&rules(&skipped, &off), "heading-order"), 0);
}

#[test]
fn test_check_widget_image_without_name_is_flagged() {
    let bare = W::new().role(AccessibleRole::Image);
    let report = A11yChecker::check_with_config(&bare, &A11yConfig::default());
    let v = report
        .violations
        .iter()
        .find(|v| v.rule == "image-alt")
        .expect("image-alt violation");
    assert_eq!((v.wcag.as_str(), v.impact), ("1.1.1", Impact::Critical));
    let named = W::new().role(AccessibleRole::Image).named("Logo");
    assert_eq!(
        count(&rules(&named, &A11yConfig::default()), "image-alt"),
        0
    );
}

#[test]
fn test_check_widget_recurses_into_children() {
    let tree = W::new().child(W::new().child(W::new().interactive()));
    let rs = rules(&tree, &A11yConfig::default());
    assert_eq!(count(&rs, "aria-label"), 1, "{rs:?}");
    assert_eq!(count(&rs, "keyboard"), 1, "{rs:?}");
}

#[test]
fn test_check_widget_focus_indicator_check_accepts_focusable() {
    let w = W::new().focusable().named("x");
    assert_eq!(count(&rules(&w, &A11yConfig::strict()), "focus-visible"), 0);
}

#[test]
fn test_to_html_attrs_empty_is_empty() {
    assert!(AriaAttributes::new().to_html_attrs().is_empty());
}

#[test]
fn test_to_html_attrs_every_field_in_order() {
    let a = AriaAttributes {
        role: Some("slider".into()),
        label: Some("Volume".into()),
        described_by: Some("hint".into()),
        hidden: true,
        expanded: Some(false),
        selected: Some(true),
        checked: Some(AriaChecked::Mixed),
        pressed: Some(AriaChecked::True),
        has_popup: Some("menu".into()),
        disabled: true,
        required: true,
        invalid: true,
        value_now: Some(5.0),
        value_min: Some(0.0),
        value_max: Some(10.0),
        value_text: Some("five".into()),
        level: Some(2),
        pos_in_set: Some(3),
        set_size: Some(9),
        controls: Some("panel".into()),
        busy: true,
        live: Some(AriaLive::Assertive),
        atomic: true,
    };
    let got = a.to_html_attrs();
    let want: Vec<(&str, &str)> = vec![
        ("role", "slider"),
        ("aria-label", "Volume"),
        ("aria-describedby", "hint"),
        ("aria-hidden", "true"),
        ("aria-expanded", "false"),
        ("aria-selected", "true"),
        ("aria-checked", "mixed"),
        ("aria-pressed", "true"),
        ("aria-haspopup", "menu"),
        ("aria-disabled", "true"),
        ("aria-required", "true"),
        ("aria-invalid", "true"),
        ("aria-valuenow", "5"),
        ("aria-valuemin", "0"),
        ("aria-valuemax", "10"),
        ("aria-valuetext", "five"),
        ("aria-level", "2"),
        ("aria-posinset", "3"),
        ("aria-setsize", "9"),
        ("aria-controls", "panel"),
        ("aria-busy", "true"),
        ("aria-live", "assertive"),
        ("aria-atomic", "true"),
    ];
    let got_ref: Vec<(&str, &str)> = got.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert_eq!(got_ref, want);
}

#[test]
fn test_to_html_attrs_false_flags_emit_nothing_and_string_escapes() {
    let a = AriaAttributes {
        hidden: false,
        disabled: false,
        label: Some("a \"b\" <c> & d".into()),
        ..AriaAttributes::new()
    };
    assert_eq!(a.to_html_attrs().len(), 1);
    assert_eq!(
        a.to_html_string(),
        "aria-label=\"a &quot;b&quot; &lt;c&gt; &amp; d\""
    );
}
