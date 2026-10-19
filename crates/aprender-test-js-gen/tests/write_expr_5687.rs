//! #5687: exact JavaScript emitted for every `Expr` variant (codegen `write_expr`).

#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)] // integration test — unwrap acceptable

use probar_js_gen::prelude::*;

fn id(name: &'static str) -> Expr {
    Expr::Ident(Identifier::new_unchecked(name))
}

fn js(e: Expr) -> String {
    generate(&JsModuleBuilder::new().expr(e).build())
        .trim_end()
        .to_string()
}

#[test]
fn literals_render_exactly_5687() {
    assert_eq!(js(Expr::Null), "null;");
    assert_eq!(js(Expr::Bool(true)), "true;");
    assert_eq!(js(Expr::Bool(false)), "false;");
    assert_eq!(js(Expr::This), "this;");
    assert_eq!(js(id("x")), "x;");
}

#[test]
fn numbers_drop_the_decimal_point_only_when_integral_5687() {
    assert_eq!(js(Expr::Num(42.0)), "42;");
    assert_eq!(js(Expr::Num(-7.0)), "-7;");
    assert_eq!(js(Expr::Num(1.5)), "1.5;");
    // Beyond i64 the integral shortcut must not truncate.
    assert_eq!(js(Expr::Num(1e30)), format!("{};", 1e30_f64));
}

#[test]
fn strings_escape_quote_backslash_and_control_chars_5687() {
    assert_eq!(js(Expr::Str("a\"b".into())), "\"a\\\"b\";");
    assert_eq!(js(Expr::Str("a\\b".into())), "\"a\\\\b\";");
    assert_eq!(js(Expr::Str("l1\nl2\r\t".into())), "\"l1\\nl2\\r\\t\";");
    assert_eq!(js(Expr::Str("plain é".into())), "\"plain é\";");
}

#[test]
fn member_index_call_new_render_exactly_5687() {
    let member = Expr::Member {
        object: Box::new(id("obj")),
        property: Identifier::new_unchecked("prop"),
    };
    assert_eq!(js(member.clone()), "obj.prop;");
    let index = Expr::Index {
        object: Box::new(id("arr")),
        index: Box::new(Expr::Num(2.0)),
    };
    assert_eq!(js(index), "arr[2];");
    let call = Expr::Call {
        callee: Box::new(member),
        args: vec![Expr::Num(1.0), id("y"), Expr::Null],
    };
    assert_eq!(js(call), "obj.prop(1, y, null);");
    let empty_call = Expr::Call {
        callee: Box::new(id("f")),
        args: vec![],
    };
    assert_eq!(js(empty_call), "f();");
    let new = Expr::New {
        constructor: Box::new(id("Map")),
        args: vec![Expr::Str("k".into())],
    };
    assert_eq!(js(new), "new Map(\"k\");");
}

#[test]
fn await_import_and_assign_render_exactly_5687() {
    assert_eq!(js(Expr::Await(Box::new(id("p")))), "await p;");
    assert_eq!(
        js(Expr::Import(Box::new(Expr::Str("./m.js".into())))),
        "import(\"./m.js\");"
    );
    let assign = Expr::Assign {
        target: Box::new(id("a")),
        value: Box::new(Expr::Num(3.0)),
    };
    assert_eq!(js(assign), "a = 3;");
}

#[test]
fn operators_are_parenthesised_and_spaced_5687() {
    let bin = |op| Expr::Binary {
        left: Box::new(id("a")),
        op,
        right: Box::new(id("b")),
    };
    assert_eq!(js(bin(BinOp::Add)), "(a + b);");
    assert_eq!(js(bin(BinOp::EqStrict)), "(a === b);");
    assert_eq!(js(bin(BinOp::Or)), "(a || b);");
    assert_eq!(js(bin(BinOp::BitAnd)), "(a & b);");
    let nested = Expr::Binary {
        left: Box::new(bin(BinOp::Mul)),
        op: BinOp::Sub,
        right: Box::new(Expr::Num(1.0)),
    };
    assert_eq!(js(nested), "((a * b) - 1);");
    let un = |op| Expr::Unary {
        op,
        operand: Box::new(id("v")),
    };
    assert_eq!(js(un(UnaryOp::Not)), "!v;");
    assert_eq!(js(un(UnaryOp::Neg)), "-v;");
    assert_eq!(js(un(UnaryOp::TypeOf)), "typeof v;");
    let ternary = Expr::Ternary {
        condition: Box::new(id("c")),
        then_expr: Box::new(Expr::Num(1.0)),
        else_expr: Box::new(Expr::Num(2.0)),
    };
    assert_eq!(js(ternary), "(c ? 1 : 2);");
}

#[test]
fn object_and_array_literals_render_exactly_5687() {
    let obj = Expr::Object(vec![("a".into(), Expr::Num(1.0)), ("b".into(), id("x"))]);
    assert_eq!(js(obj), "{ a: 1, b: x };");
    assert_eq!(js(Expr::Object(vec![])), "{  };");
    assert_eq!(
        js(Expr::Array(vec![Expr::Num(1.0), Expr::Null])),
        "[1, null];"
    );
    assert_eq!(js(Expr::Array(vec![])), "[];");
}

#[test]
fn arrows_parenthesise_params_unless_exactly_one_5687() {
    let p = |n| Identifier::new_unchecked(n);
    let one = Expr::Arrow {
        params: vec![p("x")],
        body: Box::new(id("x")),
    };
    assert_eq!(js(one), "x => x;");
    let two = Expr::Arrow {
        params: vec![p("x"), p("y")],
        body: Box::new(id("y")),
    };
    assert_eq!(js(two), "(x, y) => y;");
    let none = Expr::Arrow {
        params: vec![],
        body: Box::new(Expr::Null),
    };
    assert_eq!(js(none), "() => null;");
}

#[test]
fn arrow_block_renders_each_statement_on_its_own_line_5687() {
    let p = |n| Identifier::new_unchecked(n);
    let block = Expr::ArrowBlock {
        params: vec![p("a"), p("b")],
        body: vec![Stmt::Expr(id("a")), Stmt::Return(Some(id("b")))],
    };
    assert_eq!(js(block), "(a, b) => {\n    a;\n    return b;\n};");
    let single = Expr::ArrowBlock {
        params: vec![p("z")],
        body: vec![],
    };
    assert_eq!(js(single), "z => {\n};");
}
