#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel as PBM
from pydantic import Field


class ModelProvider(Enum):
    HF = "hf"
    LOCAL = "local"
    REPLICATE = "replicate"
    DEEPINFRA = "deepinfra"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOMAPI = "customapi"
    DUMMY = "dummy"
    VERTEXAI = "vertexai"
    GOOGLEAI = "googleai"
    TOGETHERAI = "togetherai"
    VLLM = "vllm"


class DEVICE(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class ModelType(Enum):
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ_LM = "seq2seq_lm"


class ModelConfig(PBM, arbitrary_types_allowed=True):  # type: ignore
    name: str = Field(description="Name of the model") #
    type: ModelType = Field(ModelType.CAUSAL_LM, description="Type of the model")

    answers: Optional[str] = Field(
        default=None, description="For dummy model, where to get model answers from"
    )

    provider: ModelProvider = Field(description="Provider of the model") #
    quantized: Optional[Union[bool, str]] = Field(
        default=None, description="Whether the model is quantized"
    ) #
    tokenizer_name: Optional[str] = Field(default=None, description="Name of the tokenizer") #
    dtype: str = Field(default="auto", description="Data type of the model") #
    device_map: str = Field(default="auto", description="How to map the model to different gpus")

    device: DEVICE = Field(DEVICE.CPU, description="Device to run the benchmark on") #
    batch_size: int = Field(8, description="Batch size to use for the benchmark") #

    max_batch_size: Optional[int] = Field(
        default=512, description="Maximum batch size to use for the model"
    )

    max_gen_toks: Optional[int] = Field(
        default=256, description="Maximum number of tokens to generate"
    ) #

    max_length: Optional[int] = Field(default=None, description="Maximum length of the input")
    add_special_tokens: Optional[bool] = Field(
        default=None,
        description="Whether to add special tokens to the input - None means use default (yes for Seq2Seq and no for CausalLM)",
    )

    padding_side: str = Field(default="right", description="Side to pad the input on") #  # type: ignore
    # template: Template = Field(default="{{input}}", description="Template to use for the model")
    generation_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the model on generation"
    ) #

    # Custom API specific
    url: Optional[str] = Field(default=None, description="URL (IP:Port) of the custom API")
    endpoints: Optional[Dict[str, str]] = Field(
        default=None,
        description="Endpoints of the custom API (we use generate and detect_watermark)",
    )  # NOTE Always POST with one json element "queries" of strings

    # HF Specific
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code - Only applies to HF models",
    ) #

    revision: str = Field(default="main", description="Revision of the model to use")

    subfolder: Optional[str] = Field(default=None, description="Subfolder of the model to use")
    add_generation_prompt: bool = Field(
        default=True,
        description="Whether to add a generation prompt to the end of the chat template",
    )

    seed: Optional[int] = Field(
        default=42,
        description="Seed to use for the model",
    )

    gpu_memory_utilization: float = Field(
        default=0.8,
        description="Fraction of GPU memory vLLM may reserve for the model executor",
    )
    max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum context length passed to vLLM",
    )
    max_cudagraph_capture_size: Optional[int] = Field(
        default=None,
        description="Maximum CUDA graph capture size passed to vLLM",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Whether to disable CUDA graph capture in vLLM",
    )
    language_model_only: bool = Field(
        default=False,
        description="Whether to load only the language-model path for multimodal architectures",
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="Number of GPUs to use for vLLM tensor parallelism",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Whether to disable vLLM custom all-reduce kernels",
    )
    attention_backend: Optional[str] = Field(
        default=None,
        description="Attention backend passed to vLLM, for example TRITON_ATTN",
    )
    tokenizer_mode: Optional[str] = Field(
        default=None,
        description="Tokenizer mode passed to vLLM, for example mistral",
    )
    config_format: Optional[str] = Field(
        default=None,
        description="Config format passed to vLLM, for example mistral",
    )
    load_format: Optional[str] = Field(
        default=None,
        description="Weight load format passed to vLLM, for example mistral",
    )
    use_chat_template_for_generate: Optional[bool] = Field(
        default=None,
        description="Whether plain string generation should be wrapped as a user chat message before vLLM generation",
    )
