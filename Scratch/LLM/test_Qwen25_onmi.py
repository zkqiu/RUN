
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import (
    Qwen2_5OmniModel, 
    Qwen2_5OmniProcessor, 
    AutoModelForVision2Seq, 
    AutoProcessor, 
    AutoTokenizer
)
from transformers.utils.hub import cached_file
from transformers.generation.utils import GenerateOutput

from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models._const import CPU

from datasets import load_dataset
from qwen_omni_utils import process_mm_info

class Qwen25OmniThiknerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniModel
    base_modules = [
        "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        "token2wav", 
        "thinker.audio_tower", 
        "thinker.model.rotary_emb",
        "thinker.visual", 
        "talker"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThiknerGPTQ
SUPPORTED_MODELS.append("qwen2_5_omni")

model_path = "C:/Files/Models/Qwen2.5-Omni-7B-GPTQ-4bit"

from types import MethodType

@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)

    
    model = cls._from_config(config, **kwargs)
    spk_path = cached_file(
        model_path,
        "spk_dict.pt",
        subfolder=kwargs.pop("subfolder", None),
        cache_dir=kwargs.pop("cache_dir", None),
        force_download=kwargs.pop("force_download", False),
        proxies=kwargs.pop("proxies", None),
        resume_download=kwargs.pop("resume_download", None),
        local_files_only=kwargs.pop("local_files_only", False),
        token=kwargs.pop("use_auth_token", None),
        revision=kwargs.pop("revision", None),
    )
    if spk_path is None:
        raise ValueError(f"Speaker dictionary not found at {spk_path}")
    
    model.load_speakers(spk_path)
    return model

Qwen2_5OmniModel.from_config = patched_from_config

# FP Model
# model = Qwen2_5OmniModel.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

# GPTQ MODEL
model = GPTQModel.load(
    model_path, 
    device_map="cpu", 
    torch_dtype="auto",   
    # attn_implementation="flash_attention_2"
)



from qwen_omni_utils import process_mm_info
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
# @title inference function
def inference(image_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=False, return_audio=False)

    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
image_path = "C:/Users/zhika/Downloads/1412407706563.jpeg"
prompt = "Describe this image."

# display(Video(video_path, width=640, height=360))

## Use a local HuggingFace model to inference.
response = inference(image_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print(response[0])
