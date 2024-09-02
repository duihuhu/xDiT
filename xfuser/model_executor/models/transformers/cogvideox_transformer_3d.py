from typing import Optional, Dict, Any, Union, List, Optional, Tuple, Type
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed, CogVideoXPatchEmbed

from diffusers.models import CogVideoXTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers

from xfuser.model_executor.models import xFuserModelBaseWrapper
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_world_group,
    get_cfg_group,
    get_sp_group,
    get_runtime_state, 
    initialize_runtime_state
)

from xfuser.model_executor.models.transformers.register import xFuserTransformerWrappersRegister
from xfuser.model_executor.models.transformers.base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


@xFuserTransformerWrappersRegister.register(CogVideoXTransformer3DModel)
class xFuserCogVideoXTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: CogVideoXTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d],
            submodule_name_to_wrap=["attn1"]
        )
    
    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        # print(f"device: {torch.distributed.get_rank()}: hidden_states: {hidden_states.shape}")
        
        # 3. Position embedding
        seq_length = height * width * num_frames // (self.config.patch_size**2) * get_sequence_parallel_world_size()

        pos_embeds = self.pos_embedding[:, : self.config.max_text_seq_length + seq_length]
        # print(f"device: {torch.distributed.get_rank()}: pos_embeds: {pos_embeds.shape}")
        txt_pos_embeds = pos_embeds[:, : self.config.max_text_seq_length]
        # print(f"device: {torch.distributed.get_rank()}: txt_pos_embeds: {txt_pos_embeds.shape}")
        img_pos_embeds = pos_embeds[:, self.config.max_text_seq_length \
            + get_runtime_state().pp_patches_token_start_end_idx_global[0][0]
            : self.config.max_text_seq_length + \
                get_runtime_state().pp_patches_token_start_end_idx_global[0][1]]
        # print(f"device: {torch.distributed.get_rank()}: get_runtime_state().pp_patches_token_start_end_idx_global: {get_runtime_state().pp_patches_token_start_end_idx_global}")
        # print(f"device: {torch.distributed.get_rank()}: img_pos_embeds: {img_pos_embeds.shape}")
        pos_embeds = torch.cat([txt_pos_embeds, img_pos_embeds], dim=1)
        # print(f"device: {torch.distributed.get_rank()}: pos_embeds: {pos_embeds.shape}")
        
        hidden_states = hidden_states + pos_embeds
        hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = hidden_states[:, : self.config.max_text_seq_length]
        hidden_states = hidden_states[:, self.config.max_text_seq_length :]

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                )

        hidden_states = self.norm_final(hidden_states)

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        # print(f"device: {torch.distributed.get_rank()}: hidden_states: {hidden_states.shape}")
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)