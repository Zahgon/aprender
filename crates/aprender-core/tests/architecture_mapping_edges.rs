//! Public-API coverage for `Architecture` tensor-name mapping and model-type
//! parsing (coverage ticket, Rule 7). Contract: FALSIFY-ARCH-MAP-001.

use aprender::format::converter_types::Architecture;

#[test]
fn qwen_family_maps_gguf_names_to_apr_canonical() {
    for a in [
        Architecture::Qwen2,
        Architecture::Qwen3,
        Architecture::Qwen3_5,
    ] {
        assert_eq!(
            a.map_name("blk.7.attn_q.weight"),
            "model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(
            a.map_name("blk.0.ffn_down.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
        assert_eq!(
            a.map_name("blk.3.attn_k_norm.weight"),
            "model.layers.3.self_attn.k_norm.weight"
        );
        assert_eq!(a.map_name("token_embd.weight"), "model.embed_tokens.weight");
        assert_eq!(a.map_name("output.weight"), "lm_head.weight");
        assert_eq!(a.map_name("output_norm.weight"), "model.norm.weight");
        // Unknown suffixes and names are preserved, not dropped.
        assert_eq!(
            a.map_name("blk.2.custom.weight"),
            "model.layers.2.custom.weight"
        );
        assert_eq!(a.map_name("rope_freqs.weight"), "rope_freqs.weight");
        // A "blk." name with no layer separator falls through unchanged.
        assert_eq!(a.map_name("blk.5"), "blk.5");
    }
}

#[test]
fn neox_and_opt_map_to_llama_like_names() {
    let n = Architecture::GptNeoX;
    assert_eq!(
        n.map_name("gpt_neox.layers.4.attention.query_key_value.weight"),
        "model.layers.4.self_attn.query_key_value.weight"
    );
    assert_eq!(
        n.map_name("gpt_neox.layers.4.mlp.dense_h_to_4h.bias"),
        "model.layers.4.mlp.up_proj.bias"
    );
    assert_eq!(
        n.map_name("gpt_neox.embed_in.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(n.map_name("embed_out.weight"), "lm_head.weight");
    assert_eq!(
        n.map_name("gpt_neox.final_layer_norm.bias"),
        "model.norm.bias"
    );

    let o = Architecture::Opt;
    assert_eq!(
        o.map_name("model.decoder.layers.1.self_attn.out_proj.weight"),
        "model.layers.1.self_attn.o_proj.weight"
    );
    assert_eq!(
        o.map_name("model.decoder.layers.1.fc2.bias"),
        "model.layers.1.mlp.down_proj.bias"
    );
    assert_eq!(
        o.map_name("model.decoder.embed_positions.weight"),
        "model.position_embedding.weight"
    );
    assert_eq!(
        o.map_name("model.decoder.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
}

#[test]
fn whisper_strips_model_prefix_and_passthrough_archs_keep_names() {
    for a in [Architecture::Whisper, Architecture::Moonshine] {
        assert_eq!(
            a.map_name("model.encoder.conv1.weight"),
            "encoder.conv1.weight"
        );
        assert_eq!(a.map_name("decoder.norm.weight"), "decoder.norm.weight");
    }
    for a in [
        Architecture::Auto,
        Architecture::Llama,
        Architecture::Bert,
        Architecture::Phi,
        Architecture::Mamba,
        Architecture::Rwkv7,
        Architecture::EmbeddingGemma2,
    ] {
        assert_eq!(a.map_name("model.layers.0.x"), "model.layers.0.x", "{a:?}");
    }
}

#[test]
fn model_type_parsing_is_case_insensitive_and_covers_aliases() {
    let cases = [
        ("Qwen2.5", Architecture::Qwen2),
        ("QWEN3.5", Architecture::Qwen3_5),
        ("llama3", Architecture::Llama),
        ("phi4", Architecture::Phi),
        ("pythia", Architecture::GptNeoX),
        ("galactica", Architecture::Opt),
        ("starcoder2", Architecture::Gpt2),
        ("mixtral", Architecture::Mistral),
        ("falcon3", Architecture::FalconH1),
        ("falcon40b", Architecture::FalconClassic),
        ("mamba2", Architecture::Mamba),
        ("rwkv-7", Architecture::Rwkv7),
        ("olmo2", Architecture::Llama),
        ("internlm2.5", Architecture::InternLm2),
        ("bloomz", Architecture::Bloom),
        ("gemma-embedding2", Architecture::EmbeddingGemma2),
        ("moonshine", Architecture::Moonshine),
        ("openelm", Architecture::OpenElm),
        ("deepseek-v2", Architecture::DeepSeek),
    ];
    for (s, want) in cases {
        assert_eq!(Architecture::from_model_type(s), Some(want), "{s}");
    }
    assert_eq!(Architecture::from_model_type("not-a-model"), None);
    assert_eq!(Architecture::from_model_type(""), None);
}

#[test]
fn capability_flags_and_labels_are_consistent() {
    assert!(Architecture::Llama.is_inference_verified());
    assert!(Architecture::Bert.is_inference_verified());
    assert!(!Architecture::Mamba.is_inference_verified());
    assert!(Architecture::Gpt2.is_llm() && Architecture::Bloom.is_llm());
    assert!(!Architecture::Whisper.is_llm());
    assert!(!Architecture::Bert.is_llm());
    assert!(!Architecture::Auto.is_llm());
    assert_eq!(Architecture::Qwen3_5.completeness_key(), Some("qwen3_5"));
    assert_eq!(Architecture::Phi.completeness_key(), Some("phi"));
    assert_eq!(Architecture::Gpt2.completeness_key(), None);
    assert_eq!(Architecture::Auto.display_name(), "auto-detected");
    assert_eq!(Architecture::FalconClassic.display_name(), "Falcon");
    assert_eq!(
        Architecture::EmbeddingGemma2.display_name(),
        "EmbeddingGemma 2"
    );
    assert_eq!(Architecture::default(), Architecture::Auto);
}
