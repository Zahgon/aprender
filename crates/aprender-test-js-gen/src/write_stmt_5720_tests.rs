//! #5720 — `codegen::write_stmt` exact-output tests, one per `Stmt` variant.
//!
//! The wider suite in `tests/` runs as an integration target and is not counted
//! by the `--lib` coverage run, so each arm is pinned here through the public
//! `generate` entry point. Output is compared byte for byte, indentation included.

use crate::codegen::generate;
use crate::hir::{Expr, Identifier, JsModule, Stmt};

fn id(s: &'static str) -> Identifier {
    Identifier::new_unchecked(s)
}

fn js(stmts: Vec<Stmt>) -> String {
    generate(&JsModule {
        statements: stmts,
        metadata: None,
    })
}

fn n(v: i32) -> Expr {
    Expr::Num(f64::from(v))
}

fn name(s: &'static str) -> Expr {
    Expr::Ident(id(s))
}

#[test]
fn write_stmt_5720_simple_statements() {
    let out = js(vec![
        Stmt::Let {
            name: id("a"),
            value: n(1),
        },
        Stmt::Const {
            name: id("b"),
            value: n(2),
        },
        Stmt::Assign {
            name: id("a"),
            value: n(3),
        },
        Stmt::MemberAssign {
            object: name("o"),
            member: id("m"),
            value: n(4),
        },
        Stmt::AddAssign {
            target: name("a"),
            value: n(5),
        },
        Stmt::PostIncrement(name("a")),
        Stmt::Expr(name("a")),
        Stmt::Return(None),
        Stmt::Return(Some(n(6))),
        Stmt::Comment("note".to_string()),
    ]);
    assert_eq!(
        out,
        "let a = 1;\nconst b = 2;\na = 3;\no.m = 4;\na += 5;\na++;\na;\nreturn;\nreturn 6;\n// note\n"
    );
}

#[test]
fn write_stmt_5720_if_with_and_without_else() {
    let plain = js(vec![Stmt::If {
        condition: name("c"),
        then_branch: vec![Stmt::Return(None)],
        else_branch: None,
    }]);
    assert_eq!(plain, "if (c) {\n    return;\n}\n");

    let with_else = js(vec![Stmt::If {
        condition: name("c"),
        then_branch: vec![Stmt::Return(None)],
        else_branch: Some(vec![Stmt::Expr(name("x"))]),
    }]);
    assert_eq!(with_else, "if (c) {\n    return;\n} else {\n    x;\n}\n");
}

#[test]
fn write_stmt_5720_loops_indent_their_bodies() {
    let out = js(vec![
        Stmt::For {
            var: id("i"),
            start: n(0),
            end: n(3),
            body: vec![Stmt::Expr(name("i"))],
        },
        Stmt::While {
            condition: name("go"),
            body: vec![Stmt::Expr(name("w"))],
        },
    ]);
    assert_eq!(
        out,
        "for (let i = 0; i < 3; i++) {\n    i;\n}\nwhile (go) {\n    w;\n}\n"
    );
}

#[test]
fn write_stmt_5720_try_catch_and_block_nest_one_level_deeper() {
    let out = js(vec![
        Stmt::TryCatch {
            body: vec![Stmt::Expr(name("t"))],
            catch_var: id("err"),
            handler: vec![Stmt::Expr(name("err"))],
        },
        Stmt::Block(vec![Stmt::Block(vec![Stmt::Expr(name("deep"))])]),
    ]);
    assert_eq!(
        out,
        "try {\n    t;\n} catch (err) {\n    err;\n}\n{\n    {\n        deep;\n    }\n}\n"
    );
}

#[test]
fn write_stmt_5720_worker_and_worklet_registration() {
    let out = js(vec![
        Stmt::OnMessage(vec![Stmt::Expr(name("e"))]),
        Stmt::RegisterProcessor {
            name: "proc".to_string(),
            class: id("Proc"),
        },
    ]);
    assert_eq!(
        out,
        "self.onmessage = async function(e) {\n    e;\n};\nregisterProcessor(\"proc\", Proc);\n"
    );
}
