use super::{decoder::Mixer, runtime::Qwen35};
use serde_json::{Value, json};

impl Qwen35 {
    pub(super) fn statistics(&self) -> Value {
        let mut backends = self.backends.clone();
        if let Some(mtp) = &self.mtp {
            backends.push(mtp.backend.clone());
        }
        if let Some(draft) = &self.dflash {
            backends.extend_from_slice(draft.backends());
        }
        let mut result = crate::model::metrics::backend_stats(&backends, self.head.backend());
        let mut state_bytes = 0;
        let mut verification_state_bytes = 0;
        let mut conv_bytes = 0;
        let mut log_bytes = 0;
        let mut kv_bytes = 0;
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Recurrent { state, .. } => {
                    let memory = state.memory_usage();
                    state_bytes += memory.canonical_state_bytes;
                    verification_state_bytes += memory.verification_state_bytes;
                    conv_bytes += memory.convolution_history_bytes;
                    log_bytes += memory.update_log_bytes;
                }
                Mixer::Attention { cache, .. } => {
                    kv_bytes += cache.bytes().expect("live Qwen3.5 native KV cache")
                }
            }
        }
        let fields = json!({
            "position":self.position,"layers":self.layers.len(),"architecture":self.info.architecture,
            "routed_experts":self.config.moe.as_ref().map(|m| m.experts),
            "experts_per_token":self.config.moe.as_ref().map(|m| m.top_k),
            "expert_storage":self.config.moe.as_ref().map(|_| if self.paging.is_some() {"stream"} else {"resident"}),
            "prefill_logical_chunk":self.prompt_chunk_rows(),
            "expert_cache_policy":self.paging.as_ref().map(|_| "per-layer LRU; requested and predicted protection; prompt-tail seed"),
            "expert_paging":self.paging.as_ref().map(|p|p.stats()),
            "expert_layers":self.layers.iter().filter_map(|l| match &l.ffn {super::decoder::FeedForward::Routed(m)=>Some(m.statistics()),_=>None}).collect::<Vec<_>>(),
            "model_load_ms":self.load_ms,"decoder_ms":self.forward_ms,"head_ms":self.head_ms,
            "gdn_state_bytes":state_bytes,"gdn_convolution_bytes":conv_bytes,"gdn_update_log_bytes":log_bytes,
            "gdn_verification_state_bytes":verification_state_bytes,
            "cpu_scratch_bytes":self.work.allocated_bytes()+self.logits.capacity()*4,"attention_cache_payload_bytes":kv_bytes,
            "mapped_weight_bytes":self.loader.store.mapped_bytes(),
            "attention_backend":"native-npu-gqa-stable-softmax","kv_dtype":"float16",
            "lm_head_weight_bits":self.head.bits(),"lm_head_backend":self.head.bits().map(|b| format!("open-rknpu2-w{b}a8")).unwrap_or_else(|| "cpu-floating".into()),
            "multimodal_enabled":self.loader.store.contains("model.visual.patch_embed.proj.weight"),"vision_loaded":self.vision.is_some(),"next_rope_position":self.next_rope_position,
            "vision_encoder_calls":self.vision.as_ref().map_or(0,|v|v.calls),"vision_cache_hits":self.vision.as_ref().map_or(0,|v|v.hits),
            "vision_dtype":self.options.vision_dtype.as_str(),
            "vision_encoder_ms":self.vision.as_ref().map_or(0.0,|v|v.elapsed_ms),"vision_feature_cache_bytes":self.vision.as_ref().map_or(0,|v|v.cache_bytes()),
            "prefix_truncate":"reset/recompute for shorter nonzero prefixes",
            "trace_directory":self.trace.path(),"attention_tile":self.options.attention_tile,
            "tuning_enabled":self.options.tune,"tuning_search_enabled":self.options.tune_search,"tune_cache":self.options.tune_cache,
            "activation_quantization":"full-K maxabs/127, round-away, clamp[-127,127]"
        });
        result
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let speculative = json!({
            "mtp_checkpoint_available":self.mtp_available,"speculation_enabled":self.mtp.is_some() || self.dflash.is_some(),
            "mtp_draft_tokens":self.mtp.as_ref().map_or(0,|m|m.max_draft),
            "spec_draft_n_max":crate::model::LanguageModel::speculation_window(self),
            "spec_type":if self.dflash.is_some() {self.options.speculation.kind.as_str()} else if self.mtp.is_some() {"draft-mtp"} else {"none"},
            "spec_draft_model":self.options.speculation.model,
            "dflash":self.dflash.as_ref().map(|d|d.stats()),
            "spec_draft_p_min":self.options.speculation.p_min,"spec_draft_keep_low_confidence":false,
            "spec_draft_sampling":self.options.speculation.sampling.as_str(),"spec_draft_confidence_top_k":10,
            "mtp_decoder_ms":self.mtp.as_ref().map_or(0.,|m|m.elapsed_ms),
        });
        result
            .as_object_mut()
            .unwrap()
            .extend(speculative.as_object().unwrap().clone());
        result.as_object_mut().unwrap().extend(
            serde_json::to_value(&self.phases)
                .expect("finite stage totals")
                .as_object()
                .unwrap()
                .clone(),
        );
        result.as_object_mut().unwrap().extend(
            self.work
                .recurrent
                .backend_stats()
                .as_object()
                .unwrap()
                .clone(),
        );
        result
    }
}
