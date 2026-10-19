
    // =========================================================================
    // #5684: extract_layer_generic with an in-memory TensorSource
    // =========================================================================

    struct LayerFake(std::collections::HashMap<String, Vec<f32>>);

    impl TensorSource for LayerFake {
        fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
            self.0
                .get(name)
                .cloned()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "fake".to_string(),
                    reason: format!("missing {name}"),
                })
        }
        fn has_tensor(&self, name: &str) -> bool {
            self.0.contains_key(name)
        }
        fn tensor_names(&self) -> Vec<&str> {
            self.0.keys().map(String::as_str).collect()
        }
        fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
            self.0.get(name).map(|v| vec![v.len()])
        }
    }

    /// (hf suffix, gguf suffix, fill value) for the nine mandatory layer tensors.
    const LAYER_TENSORS: [(&str, &str, f32); 9] = [
        ("input_layernorm.weight", "attn_norm.weight", 1.0),
        ("self_attn.q_proj.weight", "attn_q.weight", 2.0),
        ("self_attn.k_proj.weight", "attn_k.weight", 3.0),
        ("self_attn.v_proj.weight", "attn_v.weight", 4.0),
        ("self_attn.o_proj.weight", "attn_output.weight", 5.0),
        ("post_attention_layernorm.weight", "ffn_norm.weight", 6.0),
        ("mlp.gate_proj.weight", "ffn_gate.weight", 7.0),
        ("mlp.up_proj.weight", "ffn_up.weight", 8.0),
        ("mlp.down_proj.weight", "ffn_down.weight", 9.0),
    ];

    // hidden=8, heads=2 => head_dim=4; kv_heads=1 => kv_dim=4.
    fn layer_fake(prefix: &str, hf: bool) -> LayerFake {
        let mut m = std::collections::HashMap::new();
        for (hf_name, gguf_name, v) in LAYER_TENSORS {
            let name = if hf { hf_name } else { gguf_name };
            let len = if name.contains("q_proj") || name.contains("attn_q.") { 8 } else { 4 };
            m.insert(format!("{prefix}.{name}"), vec![v; len]);
        }
        LayerFake(m)
    }

    fn extract(src: &LayerFake) -> Result<AprTransformerLayer> {
        SafetensorsToAprConverter::extract_layer_generic(src, 3, 8, 2, 1, 16)
    }

    #[test]
    fn extract_layer_generic_hf_names_concat_qkv_in_order_5684() {
        let layer = extract(&layer_fake("model.layers.3", true)).expect("hf layer");
        let mut want = vec![2.0; 8];
        want.extend([3.0; 4]);
        want.extend([4.0; 4]);
        assert_eq!(layer.qkv_weight, want, "qkv must be [Q; K; V]");
        assert_eq!(layer.attn_norm_weight, vec![1.0; 4]);
        assert_eq!(layer.attn_output_weight, vec![5.0; 4]);
        assert_eq!(layer.ffn_norm_weight, Some(vec![6.0; 4]));
        assert_eq!(layer.ffn_gate_weight, Some(vec![7.0; 4]));
        assert_eq!(layer.ffn_up_weight, vec![8.0; 4]);
        assert_eq!(layer.ffn_down_weight, vec![9.0; 4]);
        assert!(layer.qkv_bias.is_none());
        assert!(layer.attn_q_norm_weight.is_none());
        assert!(layer.attn_k_norm_weight.is_none());
    }

    #[test]
    fn extract_layer_generic_gguf_names_match_hf_result_5684() {
        let hf = extract(&layer_fake("model.layers.3", true)).expect("hf");
        let gguf = extract(&layer_fake("blk.3", false)).expect("gguf");
        assert_eq!(hf.qkv_weight, gguf.qkv_weight);
        assert_eq!(hf.ffn_down_weight, gguf.ffn_down_weight);
        assert_eq!(hf.ffn_norm_weight, gguf.ffn_norm_weight);
    }

    #[test]
    fn extract_layer_generic_uses_layer_index_5684() {
        // Tensors exist only for layer 2; asking for layer 3 must fail.
        let err = extract(&layer_fake("model.layers.2", true));
        assert!(err.is_err(), "layer 3 must not read layer 2 tensors");
    }

    #[test]
    fn extract_layer_generic_missing_tensor_is_error_5684() {
        let mut src = layer_fake("model.layers.3", true);
        src.0.remove("model.layers.3.mlp.down_proj.weight");
        let err = extract(&src).expect_err("missing down_proj must fail");
        assert!(format!("{err}").contains("down_proj"), "got: {err}");
    }

    #[test]
    fn extract_layer_generic_qkv_bias_requires_all_three_5684() {
        let mut src = layer_fake("model.layers.3", true);
        src.0.insert("model.layers.3.self_attn.q_proj.bias".into(), vec![0.5; 8]);
        src.0.insert("model.layers.3.self_attn.k_proj.bias".into(), vec![0.6; 4]);
        assert!(extract(&src).expect("two biases").qkv_bias.is_none());
        src.0.insert("model.layers.3.self_attn.v_proj.bias".into(), vec![0.7; 4]);
        let bias = extract(&src).expect("three biases").qkv_bias.expect("bias");
        assert_eq!(bias.len(), 16);
        assert_eq!(bias[0], 0.5);
        assert_eq!(bias[8], 0.6);
        assert_eq!(bias[12], 0.7);
    }

    #[test]
    fn extract_layer_generic_qk_norm_hf_and_gguf_names_5684() {
        let mut hf = layer_fake("model.layers.3", true);
        hf.0.insert("model.layers.3.self_attn.q_norm.weight".into(), vec![0.1; 4]);
        hf.0.insert("model.layers.3.self_attn.k_norm.weight".into(), vec![0.2; 4]);
        let l = extract(&hf).expect("hf qk norm");
        assert_eq!(l.attn_q_norm_weight, Some(vec![0.1; 4]));
        assert_eq!(l.attn_k_norm_weight, Some(vec![0.2; 4]));

        let mut gg = layer_fake("blk.3", false);
        gg.0.insert("blk.3.attn_q_norm.weight".into(), vec![0.3; 4]);
        gg.0.insert("blk.3.attn_k_norm.weight".into(), vec![0.4; 4]);
        let l = extract(&gg).expect("gguf qk norm");
        assert_eq!(l.attn_q_norm_weight, Some(vec![0.3; 4]));
        assert_eq!(l.attn_k_norm_weight, Some(vec![0.4; 4]));
    }
