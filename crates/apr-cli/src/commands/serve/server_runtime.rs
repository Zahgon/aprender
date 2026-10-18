/// Run an axum server with graceful shutdown and standard banner.
/// GH-172-FIX: 16MB thread stack for worker threads — GPU forward pass with
/// large-vocab models (151K × 4B = 608KB logits) overflows the default 2MB stack.
#[cfg(feature = "inference")]
fn run_server_async(app: axum::Router, bind_addr: &str, label: &str) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(64 * 1024 * 1024) // 64 MB — GPU handler async state machines are huge
        .build()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = bind_addr.to_string();
    let label = label.to_string();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("{label} server listening on http://{bind_addr}")
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        Ok(())
    })
}

/// #5478 / FALSIFY-AUTH-001: put the Bearer gate (`APR_API_KEY_HASH` / `APR_API_KEY`)
/// in front of a router built by `create_router_with_config`. Every GGUF, CUDA, APR
/// and SafeTensors GPU router goes through this, so none serves unauthenticated while
/// a key is configured.
#[cfg(feature = "inference")]
fn gate_router(app: axum::Router) -> axum::Router {
    gate_router_with(app, super::auth::AuthGate::from_env())
}

#[cfg(feature = "inference")]
fn gate_router_with(app: axum::Router, gate: super::auth::AuthGate) -> axum::Router {
    super::auth::layer(gate, app)
}

#[cfg(all(test, feature = "inference"))]
mod gate_router_tests {
    #![allow(clippy::unwrap_used, clippy::disallowed_methods)]

    use super::super::auth::AuthGate;
    use super::gate_router_with;
    use axum::body::Body;
    use axum::http::{header, Request, StatusCode};
    use tower::ServiceExt as _;

    async fn status(gate: AuthGate, bearer: Option<&str>) -> StatusCode {
        let app = gate_router_with(
            axum::Router::new().route("/health", axum::routing::get(|| async { "ok" })),
            gate,
        );
        let mut req = Request::builder().uri("/health");
        if let Some(b) = bearer {
            req = req.header(header::AUTHORIZATION, b);
        }
        app.oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn gated_router_is_401_without_key_and_200_with_it() {
        let k = || AuthGate::from_plain_key("k");
        assert_eq!(status(k(), None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(status(k(), Some("Bearer wrong")).await, StatusCode::UNAUTHORIZED);
        assert_eq!(status(k(), Some("Bearer k")).await, StatusCode::OK);
        assert_eq!(status(AuthGate::disabled(), None).await, StatusCode::OK);
    }

    /// Structural guard: every `create_router_with_config(` call in the serve
    /// sources is wrapped by `gate_router(`. A new router that skips the gate
    /// turns this RED (#5478).
    #[test]
    fn every_realizar_router_is_wrapped_by_gate_router() {
        let sources = [
            ("server.rs", include_str!("server.rs")),
            ("handler_gpu_completion.rs", include_str!("handler_gpu_completion.rs")),
            ("handlers_include_01.rs", include_str!("handlers_include_01.rs")),
        ];
        for (name, src) in sources {
            for (i, line) in src.lines().enumerate() {
                let t = line.trim_start();
                if t.contains("create_router_with_config(")
                    && !t.starts_with("use ")
                    && !t.starts_with("//")
                    && !t.contains("gate_router(")
                    && !t.contains("let router = realizar::api::create_router_with_config")
                {
                    panic!("{name}:{}: ungated router: {t}", i + 1);
                }
            }
        }
    }
}
