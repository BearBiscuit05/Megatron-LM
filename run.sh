set -x
  # 128 Nodes
  # MODEL=DeepSeek-V3 PP=16 VPP=1 TP=1 EP=64 RUN_TIME=04:00:00 NNODES=128 GBS=8192 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj layernorm moe
  # MODEL=DeepSeek-V3 PP=16 VPP=1 TP=2 EP=64 NNODES=128 GBS=8192 RUN_TIME=04:00:00 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj layernorm --moe-track-imbalance-rate
  # MODEL=DeepSeek-V3 PP=16 VPP=1 TP=2 EP=64 NNODES=128 GBS=8192 RUN_TIME=00:20:00 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj layernorm --moe-track-imbalance-rate --moe-router-force-load-balancing

  # 64 Nodes
  # MODEL=DeepSeek-V3 PP=8 VPP=1 TP=4 EP=32 PP_FIRST=8 PP_LAST=5 NNODES=64 GBS=8192 RUN_TIME=00:40:00 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj layernorm --moe-track-imbalance-rate --moe-router-force-load-balancing

export SEED_MODELS_LOGGING_LEVEL=WARN
export OMNISTORE_LOGGING_LEVEL=ERROR
export BPEX_NO_WARN_ON_UNTUNED_CASE=1
export TOKENIZERS_PARALLELISM=false
export VESCALE_SINGLE_DEVICE_RAND=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}



DISTRIBUTED_ARGS=(
    --nproc_per_node $NPROC_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)



MODEL_ARGS=(
  # Distributed args
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 16
  --decoder-first-pipeline-num-layers 9
  --decoder-last-pipeline-num-layers 10
  --expert-model-parallel-size 8
  --context-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather

  # Training args
  --use-mcore-models
  --sequence-parallel
  --use-flash-attn
  --disable-bias-linear
  --micro-batch-size 1
  --global-batch-size 8192
  --train-samples 585937500
  --exit-duration-in-mins 220
  --no-check-for-nan-in-loss-and-grad
  --no-rope-fusion
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl te # TODO This is an EA feature, only available in te 2.2
  #--mla-yarn-rope-fusion # TODO This is an EA feature
  --disable-bf16-reduced-precision-matmul
  --recompute-granularity selective
  --recompute-modules mla_up_proj layernorm

  # Transformer Engine args
  --transformer-impl transformer_engine

  # Data args
  --seq-length 4096
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model deepseek-ai/DeepSeek-V3
  --data-path /mnt/hdfs/xya/data/megatron_train/meg-gpt2_text_document
  --split 99,1,0
  --no-mmap-bin-files
  --no-create-attention-mask-in-dataloader
  --num-workers 6

  # Add network size args
  --num-layers 61
  --hidden-size 7168
  --ffn-hidden-size 18432
  --num-attention-heads 128
  --kv-channels 128
  --max-position-embeddings 4096
  --position-embedding-type rope
  --rotary-base 10000
  --make-vocab-size-divisible-by 3232
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --swiglu
  --untie-embeddings-and-output-weights
  --multi-latent-attention
  # Comment out the following MTP args to disable MTP
  --mtp-num-layers 1
  --mtp-loss-scaling-factor 0.1

  # Add regularization args
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --clip-grad 1.0
  --weight-decay 0.1
  --qk-layernorm

  # Add learning rate args
  --lr-decay-samples 584765624
  --lr-warmup-samples 1536000
  # Learning rate scaled down from 7.3e-6 (DeepSeek-V3 technical report, GBS=15360) to 3.9e-6 (GBS=8192)
  --lr-warmup-init 3.9e-7
  --lr 3.9e-6
  --min-lr 3.9e-7
  --lr-decay-style cosine
  --adam-beta1 0.9
  --adam-beta2 0.95

  # Add MoE args
  --num-experts 256
  --moe-layer-freq "([0]*3+[1]*58)"
  --moe-ffn-hidden-size 2048
  --moe-shared-expert-intermediate-size 2048
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk 8
  --moe-token-dispatcher-type alltoall
  --moe-router-pre-softmax
  --moe-grouped-gemm
  --moe-aux-loss-coeff 1e-4
  --moe-router-group-topk 4
  --moe-router-num-groups 8
  --moe-router-topk-scaling-factor 2.5
  --moe-router-score-function sigmoid
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 1e-3
  --moe-router-dtype fp32
  --moe-permute-fusion
  #--moe-track-imbalance-rate 
  #--moe-router-force-load-balancing
  # Add MLA args
  --q-lora-rank 1536
  --kv-lora-rank 512
  --qk-head-dim 128
  --qk-pos-emb-head-dim 64
  --v-head-dim 128
  --rotary-scaling-factor 40
  --mscale 1.0
  --mscale-all-dim 1.0

  # Add validation args
  --eval-iters 32
  --eval-interval 200

  # Add checkpointing args
  --finetune
  --auto-detect-ckpt-format
  
  --save-interval 500
  --dist-ckpt-strictness log_all

  # Add initialization args
  --init-method-std 0.02

  # Add logging args
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-num-zeros-in-grad
  --log-params-norm
  --log-validation-ppl-to-tensorboard
  --log-throughput
  --log-interval 1
  --logging-level 40
#   --tensorboard-dir ${OUTPUT_PATH}/tensorboard
#   --wandb-project ${WANDB_PROJECT}
#   --wandb-exp-name DeepSeek-V3-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}

  --bf16
  --load /mnt/hdfs/xya/mega_model/dpsk-671b-bf16
  --save /mnt/hdfs/xya/mega_model/save_disk_ckpt_dpsk_671b_bf16
  --ckpt-convert-save /mnt/hdfs/xya/mega_model/disk_ckpt_dpsk_671b_bf16
  --ckpt-convert-format torch_dist
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]}