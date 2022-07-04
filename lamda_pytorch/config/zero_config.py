from colossalai.zero.shard_utils import TensorShardStrategy

zero = dict(
    model_config = dict(
        shard_strategy = TensorShardStrategy(),
        tensor_placement_policy = 'cpu',
        reuse_fp16_shard = False
    )
)

gradient_accumulation = 4
clip_grad_norm = 1.0