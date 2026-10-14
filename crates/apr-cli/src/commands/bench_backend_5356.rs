    // ========================================================================
    // G1.3 (#5356): `apr bench --backend` device choice
    // ========================================================================

    #[test]
    fn bench_without_backend_keeps_the_automatic_choice() {
        assert_eq!(bench_cuda_choice(None, true).expect("auto"), (true, false));
        assert_eq!(bench_cuda_choice(None, false).expect("auto"), (false, false));
    }

    #[test]
    fn bench_backend_cpu_measures_cpu_even_with_a_gpu() {
        assert_eq!(bench_cuda_choice(Some("cpu"), true).expect("cpu"), (false, false));
    }

    #[test]
    fn bench_backend_cuda_with_a_device_is_required_not_preferred() {
        assert_eq!(bench_cuda_choice(Some("cuda"), true).expect("cuda"), (true, true));
    }

    #[test]
    fn bench_backend_cuda_without_a_device_is_refused_not_downgraded() {
        let err = bench_cuda_choice(Some("cuda"), false).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("--backend cuda"), "names the flag typed: {msg}");
        assert!(msg.contains("--backend cpu"), "names the deliberate CPU route: {msg}");
    }

    #[test]
    fn bench_backend_wgpu_is_refused_because_bench_has_no_wgpu_path() {
        for ready in [true, false] {
            let err = bench_cuda_choice(Some("wgpu"), ready).expect_err("must refuse");
            assert!(err.to_string().contains("--backend wgpu"), "{err}");
        }
    }

    #[test]
    fn bench_backend_beside_brick_is_refused() {
        let err = run(
            Path::new("model.gguf"),
            0,
            1,
            1,
            None,
            false,
            Some("rms_norm"),
            true,
            &[50.0],
            Some("cpu"),
        )
        .expect_err("--backend with --brick must refuse");
        assert!(err.to_string().contains("--brick"), "{err}");
    }
