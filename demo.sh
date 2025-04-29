num_layers: 61 # for original V3 = 61
      hidden_size: 7168
      ffn_hidden_size: 18432
      num_attention_heads: 128
      original_max_position_embeddings: 4096
      seq_length: 2048
      vocab_size: 129280 # 129280 + extra_vocab_size - 18?
      position_embedding_type: rope
      apply_rope_fusion: false
      qk_layernorm: true
      rotary_base: 10000
      swiglu: true
      untie_embeddings_and_output_weights: true
      normalization: RMSNorm
      norm_epsilon: 1.0e-06
      add_bias_linear: false
      # moe config
      moe_layer_freq: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # first one is dense and others are moe layer, layer_num is 27
      decoder_first_pipeline_num_layers: 4
      decoder_last_pipeline_num_layers: 1 
      num_experts: 256
      moe_ffn_hidden_size: 2048
      enable_shared_expert: true
      moe_shared_expert_intermediate_size: 2048 # only one shared experts
      moe_grouped_gemm: true
      moe_token_dispatcher_type: alltoall
      moe_router_topk: 8
      moe_router_load_balancing_type: seq_aux_loss
      moe_aux_loss_coeff: 0.0001
      moe_router_num_groups: 8
      moe_router_group_topk: 4
      moe_router_dtype: fp32
      moe_router_enable_expert_bias: true
      moe_router_bias_update_rate: 1e-3
      moe_router_pre_softmax: true
      moe_router_topk_scaling_factor: 2.5
      moe_router_score_function: sigmoid
      q_lora_rank: 1536
      kv_lora_rank: 512
      qk_head_dim: 128
      qk_pos_emb_head_dim: 64
      v_head_dim: 128
      transformer_impl: transformer_engine
      attention_dropout: 0
      hidden_dropout: 0
      multi_latent_attention: true
      # training config
      micro_batch_size: 1
      global_batch_size: 1536 # 16*4
      train_iters: 800
      weight_decay: 0.1
      adam_beta1: 0.9
      adam_beta2: 0.95
      init_method_std: 0.02
      clip_grad: 1.0
      bf16: true
      lr: 5.0e-7
      lr_warmup_iters: 200
      min_lr: 0
      seed: 42
      # 3D parallel
      attention_backend: flash
      expert_model_parallel_size: 16
      pipeline_model_parallel_size: 16
      tensor_model_parallel_size: 1
      # context-parallel-size: 2
      sequence_parallel: false
      # full ac
      recompute_method: uniform
      recompute_num_layers: 1
      recompute_granularity: full
      # opt offload
      # optimizer-cpu-offload: true
      # use-precision-aware-optimizer: true
      # optimizer-offload-fraction: true
      use_distributed_optimizer: true
      overlap_param_gather: true
      overlap_grad_reduce: true
      tokenizer_type: "HuggingFaceTokenizer"
      no_masked_softmax_fusion: true
      attention_softmax_in_fp32: true