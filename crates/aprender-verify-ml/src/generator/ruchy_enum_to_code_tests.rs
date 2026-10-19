//! Branch tests for `RuchyNode::to_code` / `RuchyPattern::to_code` (#5682).

use super::ruchy_enum::{
    RuchyBinaryOp, RuchyCompareOp, RuchyNode as N, RuchyPattern, RuchyType, RuchyUnaryOp,
};

fn id(s: &str) -> N {
    N::Ident(s.into())
}
fn b(n: N) -> Box<N> {
    Box::new(n)
}

#[test]
fn module_joins_items_with_blank_line() {
    let m = N::Module(vec![N::Break, N::Continue]);
    assert_eq!(m.to_code(0), "break;\n\ncontinue;");
}

#[test]
fn fn_def_renders_params_return_and_indented_body() {
    let f = N::FnDef {
        name: "f".into(),
        params: vec![("a".into(), RuchyType::I32), ("c".into(), RuchyType::Bool)],
        return_type: Some(RuchyType::Option(Box::new(RuchyType::Vec(Box::new(
            RuchyType::Custom("T".into()),
        ))))),
        body: vec![N::Return(Some(b(N::IntLit(1))))],
    };
    assert_eq!(
        f.to_code(1),
        "    fn f(a: i32, c: bool) -> Option<Vec<T>> {\n        return 1;\n    }"
    );
}

#[test]
fn assign_binop_unary_compare_nullcoalesce() {
    let a = N::Assign {
        target: b(id("x")),
        value: b(N::IntLit(2)),
    };
    assert_eq!(a.to_code(0), "x = 2;");
    let bo = N::BinOp {
        left: b(id("x")),
        op: RuchyBinaryOp::BitXor,
        right: b(N::IntLit(3)),
    };
    assert_eq!(bo.to_code(0), "(x ^ 3)");
    for (op, s) in [
        (RuchyUnaryOp::Neg, "-"),
        (RuchyUnaryOp::Not, "!"),
        (RuchyUnaryOp::Ref, "&"),
        (RuchyUnaryOp::RefMut, "&mut "),
        (RuchyUnaryOp::Deref, "*"),
    ] {
        let u = N::UnaryOp {
            op,
            operand: b(id("x")),
        };
        assert_eq!(u.to_code(0), format!("({s}x)"));
    }
    let c = N::Compare {
        left: b(id("x")),
        op: RuchyCompareOp::Ge,
        right: b(N::IntLit(0)),
    };
    assert_eq!(c.to_code(0), "(x >= 0)");
    let n = N::NullCoalesce {
        value: b(id("x")),
        default: b(N::IntLit(0)),
    };
    assert_eq!(n.to_code(0), "x ?? 0");
}

#[test]
fn literals_render_exactly() {
    assert_eq!(N::FloatLit(2.0).to_code(0), "2.0");
    assert_eq!(N::StrLit("hi".into()).to_code(0), "\"hi\"");
    assert_eq!(N::BoolLit(false).to_code(0), "false");
    assert_eq!(N::IntLit(-5).to_code(0), "-5");
}

#[test]
fn if_without_and_with_else() {
    let no_else = N::If {
        cond: b(id("c")),
        then_body: vec![N::Break],
        else_body: vec![],
    };
    assert_eq!(no_else.to_code(0), "if c {\n    break;\n}");
    let with_else = N::If {
        cond: b(id("c")),
        then_body: vec![N::Break],
        else_body: vec![N::Continue],
    };
    assert_eq!(
        with_else.to_code(0),
        "if c {\n    break;\n} else {\n    continue;\n}"
    );
}

#[test]
fn match_renders_arms_with_patterns() {
    let m = N::Match {
        value: b(id("v")),
        arms: vec![
            (
                N::Pattern(RuchyPattern::Wildcard),
                vec![N::Break, N::Continue],
            ),
            (
                N::Pattern(RuchyPattern::Tuple(vec![
                    RuchyPattern::Binding("a".into()),
                    RuchyPattern::Literal(b(N::IntLit(1))),
                ])),
                vec![],
            ),
        ],
    };
    assert_eq!(
        m.to_code(0),
        "match v {\n    _ => { break;; continue; },\n    (a, 1) => {  }\n}"
    );
}

#[test]
fn loops_and_return_forms() {
    let w = N::While {
        cond: b(N::BoolLit(true)),
        body: vec![N::Break],
    };
    assert_eq!(w.to_code(0), "while true {\n    break;\n}");
    assert_eq!(N::Return(None).to_code(0), "return;");
    assert_eq!(N::Return(None).to_code(2), "        return;");
}

#[test]
fn calls_and_member_access() {
    let c = N::Call {
        func: b(id("f")),
        args: vec![N::IntLit(1), id("y")],
    };
    assert_eq!(c.to_code(0), "f(1, y)");
    let m = N::MethodCall {
        receiver: b(id("o")),
        method: "m".into(),
        args: vec![],
    };
    assert_eq!(m.to_code(0), "o.m()");
    let fa = N::FieldAccess {
        receiver: b(id("o")),
        field: "f".into(),
    };
    assert_eq!(fa.to_code(0), "o.f");
    let oc = N::OptionalChain {
        receiver: b(id("o")),
        field: "f".into(),
    };
    assert_eq!(oc.to_code(0), "o?.f");
}

#[test]
fn struct_def_and_init() {
    let d = N::StructDef {
        name: "P".into(),
        fields: vec![
            ("x".into(), RuchyType::F64),
            ("n".into(), RuchyType::String),
        ],
    };
    assert_eq!(d.to_code(0), "struct P {\n    x: f64,\n    n: String,\n}");
    let i = N::StructInit {
        name: "P".into(),
        fields: vec![
            ("x".into(), N::FloatLit(1.0)),
            ("n".into(), N::StrLit("a".into())),
        ],
    };
    assert_eq!(i.to_code(0), "P { x: 1.0, n: \"a\" }");
}

#[test]
fn array_closure_block_spawn_send() {
    assert_eq!(
        N::Array(vec![N::IntLit(1), N::IntLit(2)]).to_code(0),
        "[1, 2]"
    );
    let cl = N::Closure {
        params: vec!["a".into(), "c".into()],
        body: b(id("a")),
    };
    assert_eq!(cl.to_code(0), "|a, c| a");
    assert_eq!(
        N::Block(vec![N::Break]).to_code(1),
        "    {\n        break;\n    }"
    );
    assert_eq!(N::Spawn(b(id("t"))).to_code(0), "spawn t");
    let s = N::Send {
        target: b(id("a")),
        message: b(N::IntLit(1)),
    };
    assert_eq!(s.to_code(0), "a <- 1");
}
