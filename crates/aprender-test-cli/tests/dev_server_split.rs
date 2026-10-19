//! `DevServer::run_split` — HTTP and WebSocket on separate ports.
//!
//! Contract: `contracts/test-cli-dev-server-run-split-v1.yaml`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use futures::StreamExt;
use probador::{DevServer, DevServerConfig, HotReloadMessage};
use std::path::Path;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe");
    l.local_addr().expect("addr").port()
}

fn config(dir: &Path, cors: bool) -> DevServerConfig {
    DevServerConfig {
        directory: dir.to_path_buf(),
        port: free_port(),
        ws_port: free_port(),
        cors,
        cross_origin_isolated: false,
    }
}

/// Connect, retrying until the spawned server has bound the port.
async fn connect(port: u16) -> TcpStream {
    for _ in 0..100 {
        if let Ok(s) = TcpStream::connect(("127.0.0.1", port)).await {
            return s;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server never bound port {port}");
}

async fn http_get(port: u16, path: &str) -> String {
    let mut s = connect(port).await;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: x\r\nOrigin: http://o.example\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).await.expect("write");
    let mut out = Vec::new();
    s.read_to_end(&mut out).await.expect("read");
    String::from_utf8_lossy(&out).into_owned()
}

fn site() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    std::fs::write(d.path().join("index.html"), "<h1>split-index</h1>").expect("index");
    std::fs::write(d.path().join("app.js"), "let split_js = 1;").expect("js");
    d
}

#[tokio::test]
async fn split_serves_index_and_static_files_on_http_port() {
    let dir = site();
    let cfg = config(dir.path(), false);
    let port = cfg.port;
    let task = tokio::spawn(async move { DevServer::new(cfg).run_split().await });
    let root = http_get(port, "/").await;
    assert!(root.starts_with("HTTP/1.1 200"), "{root}");
    assert!(root.contains("<h1>split-index</h1>"), "{root}");
    let js = http_get(port, "/app.js").await;
    assert!(js.contains("let split_js = 1;"), "{js}");
    let missing = http_get(port, "/nope.txt").await;
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
    task.abort();
}

#[tokio::test]
async fn split_cors_header_follows_config() {
    for (cors, expect) in [(true, true), (false, false)] {
        let dir = site();
        let cfg = config(dir.path(), cors);
        let port = cfg.port;
        let task = tokio::spawn(async move { DevServer::new(cfg).run_split().await });
        let resp = http_get(port, "/").await.to_lowercase();
        assert_eq!(
            resp.contains("access-control-allow-origin"),
            expect,
            "cors={cors}: {resp}"
        );
        task.abort();
    }
}

#[tokio::test]
async fn split_websocket_port_sends_server_ready_and_forwards_broadcasts() {
    let dir = site();
    let cfg = config(dir.path(), false);
    let ws_port = cfg.ws_port;
    let server = DevServer::new(cfg);
    let tx = server.reload_sender();
    let task = tokio::spawn(async move { server.run_split().await });
    drop(connect(ws_port).await);
    let url = format!("ws://127.0.0.1:{ws_port}/");
    let (mut ws, _) = tokio_tungstenite::connect_async(url)
        .await
        .expect("ws connect");
    let first = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout")
        .expect("closed")
        .expect("frame");
    assert_eq!(
        first.into_text().expect("text"),
        HotReloadMessage::ServerReady.to_json()
    );
    tx.send(HotReloadMessage::RebuildStarted)
        .expect("subscriber present");
    let second = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .expect("timeout")
        .expect("closed")
        .expect("frame");
    assert_eq!(
        second.into_text().expect("text"),
        HotReloadMessage::RebuildStarted.to_json()
    );
    task.abort();
}

#[tokio::test]
async fn split_http_port_does_not_speak_websocket_and_ws_port_does_not_serve_files() {
    let dir = site();
    let cfg = config(dir.path(), false);
    let (port, ws_port) = (cfg.port, cfg.ws_port);
    let task = tokio::spawn(async move { DevServer::new(cfg).run_split().await });
    drop(connect(port).await);
    let on_ws = http_get(ws_port, "/app.js").await;
    assert!(
        !on_ws.contains("split_js"),
        "ws port leaked static file: {on_ws}"
    );
    task.abort();
}

#[tokio::test]
async fn split_returns_addr_in_use_when_http_port_is_taken() {
    let dir = site();
    let cfg = config(dir.path(), false);
    let held = std::net::TcpListener::bind(("0.0.0.0", cfg.port)).expect("hold port");
    let err = tokio::time::timeout(Duration::from_secs(5), DevServer::new(cfg).run_split())
        .await
        .expect("run_split must return, not serve")
        .expect_err("bind must fail");
    assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse);
    drop(held);
}

#[tokio::test]
async fn split_returns_addr_in_use_when_ws_port_is_taken() {
    let dir = site();
    let cfg = config(dir.path(), false);
    let held = std::net::TcpListener::bind(("0.0.0.0", cfg.ws_port)).expect("hold port");
    let err = tokio::time::timeout(Duration::from_secs(5), DevServer::new(cfg).run_split())
        .await
        .expect("run_split must return, not serve")
        .expect_err("bind must fail");
    assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse);
    drop(held);
}
