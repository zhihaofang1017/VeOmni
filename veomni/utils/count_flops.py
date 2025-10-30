# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers import PretrainedConfig

from . import logging
from .device import get_device_name


logger = logging.get_logger(__name__)


def get_device_flops(unit="T"):
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    elif "B200" in device_name:
        flops = 2250e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class VeomniFlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = VeomniFlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(batch_seqlens, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        self.estimate_func = {
            "qwen2_vl": self._estimate_qwen2_vl_flops,
            # the only difference between Qwen2 and Qwen2.5 for counting flops is the window attention
            # used in the ViT for Qwen2.5VL which is considered in the _estimate_qwen2_vl_flops function.
            "qwen2_5_vl": self._estimate_qwen2_vl_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "qwen3_moe": self._estimate_qwen3_moe_flops,
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_qwen2_flops,
            # qwen3 reused _estimate_qwen2_flops func because the only model structure diff between qwen2 dense and qwen3 dense is that
            # qwen3 has additional RMSNorm layers for q and k.
            # RMSNorm layers have minimal impact at the MFU and can be ignored.
            "qwen3": self._estimate_qwen2_flops,
            "seed_oss": self._estimate_seed_flops,
        }

        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        return 0

    def _estimate_seed_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts
        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts
        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank
        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + emd_and_lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum
        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers
        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_num_expert = self.config.num_experts
        moe_topk = self.config.num_experts_per_tok

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk) * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers) + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_llama_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_vl_flops(self, tokens_sum, batch_seqlens, delta_time, **kargs):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # vit flops
        image_seqlens = kargs.get("image_seqlens", None)
        if image_seqlens is not None:
            vit_flops = self._estimate_qwen_vit_flop(image_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen_vit_flop(self, image_seqlens, config):
        """
        Estimate the FLOPS of the vision encoder for Qwen2 and Qwen2.5
        """

        if config is None:
            return 0
        tokens_sum = sum(image_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        # In Qwen2 VL and Qwen2.5VL, the parameters naming are different:
        #
        # Parameter                 | Qwen2 VL         | Qwen2.5 VL
        # --------------------------|------------------|------------------
        # ViT hidden dimension      | embed_dim        | hidden_size
        # ViT output dimension      | hidden_size      | out_hidden_size
        # ViT MLP intermediate dim  | embed_dim * mlp_ratio | intermediate_size
        #
        # See https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
        # and https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json for an example.
        is_qwen2_vl = hasattr(config, "embed_dim")
        dim = config.embed_dim if is_qwen2_vl else config.hidden_size
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio) if is_qwen2_vl else config.intermediate_size
        out_hidden_size = config.hidden_size if is_qwen2_vl else config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size
        head_dim = dim // num_heads

        # Qwen 2.5 VL uses SiLU, thus 3.
        mlp_N = dim * mlp_hidden_dim * (2 if is_qwen2_vl else 3)
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        patch_embed_and_merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (
            dim * (spatial_merge_size**2)
        )

        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * depth + patch_embed_and_merger_N

        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # In Qwen2.5 VL, windowed attention is used in some layers.
        full_attn_layer_num = config.depth if is_qwen2_vl else len(config.fullatt_block_indexes)
        window_attn_layer_num = config.depth - full_attn_layer_num

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in image_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        # If window attention is used, add the window attention flops
        if window_attn_layer_num > 0:
            window_attn_compute_flops = 12 * tokens_sum * (config.window_size**2) * head_dim * num_heads
            attn_qkv_flops += window_attn_compute_flops * window_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    def estimate_flops(self, batch_seqlens, delta_time, **kwargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time, **kwargs)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
