#    Copyright 2026 Federico Marcuzzi, INSAIT
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


from src.models.base.utils_vllm import *

from collections.abc import Mapping
from numbers import Integral

import torch
import numpy as np

from .base_model import BaseModel
from src.configs.base_model_config import DEVICE, ModelProvider

from transformers import AutoTokenizer
import vllm
from vllm import LLM, SamplingParams


class VLLMCausalLM(BaseModel):
    AUTO_TOKENIZER_CLASS = AutoTokenizer
    AUTO_MODEL_CLASS = LLM

    def __init__(self, config):
        print("[INFO] vllm version:", vllm.__version__)
        super().__init__(config)

        self._batch_size = config.batch_size
        self._max_gen_toks = config.max_gen_toks
        self._quantized = config.quantized
        self._device = ("cuda" if config.device in [DEVICE.AUTO, DEVICE.CUDA] and torch.cuda.is_available() else config.device.value)
        self._generation_args = config.generation_args
        self._dtype = config.dtype
        self._provider = config.provider
        self._seed = config.seed
        self._trust_remote_code = config.trust_remote_code
        self._padding_side = config.padding_side
        self.model_path = config.name
        self.tokenizer_path = config.tokenizer_name

        self._gpu_memory_utilization = getattr(config, "gpu_memory_utilization", 0.8)
        self._max_model_len = getattr(config, "max_model_len", None)
        self._max_cudagraph_capture_size = getattr(config, "max_cudagraph_capture_size", None)
        self._enforce_eager = getattr(config, "enforce_eager", False)
        self._language_model_only = getattr(config, "language_model_only", False)
        self._tensor_parallel_size = getattr(config, "tensor_parallel_size", None)
        self._disable_custom_all_reduce = getattr(config, "disable_custom_all_reduce", False)
        self._attention_backend = getattr(config, "attention_backend", None)
        self._tokenizer_mode = getattr(config, "tokenizer_mode", None)
        self._config_format = getattr(config, "config_format", None)
        self._load_format = getattr(config, "load_format", None)
        self._use_chat_template_for_generate = getattr(config, "use_chat_template_for_generate", None)

        assert (self._provider == ModelProvider.VLLM)

        self.kvc_allowed_quant = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
        print(f"[INFO] Model path: {self.model_path}")

        if self._device == "cuda":
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = 1
        if self._tensor_parallel_size is None:
            self.tensor_parallel_size = self.num_gpus
        else:
            self.tensor_parallel_size = max(1, min(int(self._tensor_parallel_size), self.num_gpus))
        print(f"[INFO] Using tensor_parallel_size={self.tensor_parallel_size} of {self.num_gpus} visible GPUs for VLLM model: {self.model_path}")

        self.model = self._load_model()
        self._max_length = self.model.llm_engine.model_config.max_model_len

        self.tokenizer = self._load_tokenizer()
        self.tokenizer.model_max_length = self._max_length

        self.sampling_params = self._set_sampling_params(config.generation_args)
        self.num_pmt_toks, self.num_gen_toks, self.num_tot_prmt, self.num_tot_gens = 0, 0, 0, 0
    
    # TO implement
    def loglikelihood(self, prompts):
        return self._loglikelihood(prompts)[0]
    
    # TO implement
    def perplexities(self, prompts):
        return self._loglikelihood(prompts)[1]

    # TO implement
    def most_prob_options(self, prompts, anwers, get_soft_max=True, n=1):
        if VLLM_VERSION <= Version("0.7.2"):
            get_logits = VLLMLogitsRetriever(self.tokenizer, anwers)
            temp_sampling_params = self._get_temp_updated_sampling_params({"max_tokens" : 1, "detokenize" : True, "logits_processors" : [get_logits]})
            outputs = self.model.generate(prompts, sampling_params=temp_sampling_params)
        else:
            get_logits = VLLMLogitsRetrieverNew(self.tokenizer, anwers, prompts)
            list_sp = get_logits.get_sample_params_list(self._generation_args, {"max_tokens" : 1, "detokenize" : True})
            outputs = self.model.generate(prompts, sampling_params=list_sp)
            get_logits.compute_data()        
    
        for prompts in outputs:
            # statistics:
            self.num_pmt_toks += len(prompts.prompt_token_ids)
            self.num_gen_toks += 1 # here the model generates only one token per prompt.
            self.num_tot_prmt += 1
            self.num_tot_gens += 1

        if get_soft_max:
            outputs = get_logits.get_soft_max()
        else:
            outputs = get_logits.get_all()

        return outputs

    # TO implement
    def generate(self, input, n=1, max_tokens=None):
        if self._should_use_chat_template_for_generate():
            chats = [
                [{"role": "user", "content": str(prompt)}]
                for prompt in self._as_prompt_list(input)
            ]
            return self.generate_chat(
                chats,
                add_generation_prompt=True,
                continue_final_message=False,
                n=n,
                max_tokens=max_tokens,
            )

        flag_regroup = False
        if n > 1:
            input = np.repeat(input, n)
            flag_regroup = True

        max_tokens = max_tokens if max_tokens is not None and max_tokens > 0 else self._max_gen_toks
        temp_sampling_params = self._get_temp_updated_sampling_params({"max_tokens" : max_tokens})
        outputs = self.model.generate(input, sampling_params=temp_sampling_params)
        return self._extract_generation_text(outputs, flag_regroup, n)

    @staticmethod
    def _as_prompt_list(input):
        if isinstance(input, str):
            return [input]
        if hasattr(input, "tolist"):
            return input.tolist()
        return list(input)

    def _should_use_chat_template_for_generate(self):
        if self._use_chat_template_for_generate is not None:
            return bool(self._use_chat_template_for_generate)

        model_text = f"{self.model_path} {self.tokenizer_path or ''}".lower()
        return "eurollm" in model_text and getattr(self.tokenizer, "chat_template", None) is not None

    def _extract_generation_text(self, outputs, flag_regroup=False, n=1):
        if not outputs:
            return []

        list_output = []
        self.num_tot_prmt += len(outputs)
        for prompts in outputs:
            self.num_pmt_toks += len(prompts.prompt_token_ids)
            self.num_tot_gens += len(prompts.outputs)
            list_gen = []
            for gen in prompts.outputs:
                self.num_gen_toks += len(gen.token_ids)
                list_gen.append(gen.text)
            
            list_output.append(list_gen)

        if len(list_output[0]) == 1:
            list_output = [gen[0] for gen in list_output]

        if flag_regroup:
            list_output = np.array(list_output).reshape(-1, n).tolist()

        return list_output

    # TO implement
    def generate_chat(self, chats, add_generation_prompt=False, continue_final_message=True, n=1, max_tokens=None, *args, **kwargs):
        flag_regroup = False
        if n > 1:
            repeated_chats = []
            for chat in chats:
                repeated_chats.extend([chat] * n)
            chats = repeated_chats
            flag_regroup = True

        max_tokens = max_tokens if max_tokens is not None and max_tokens > 0 else self._max_gen_toks
        temp_sampling_params = self._get_temp_updated_sampling_params({"max_tokens" : max_tokens})

        if not self._can_use_chat_template():
            rendered_chats = self._render_chats_without_template(chats, continue_final_message)
            outputs = self.model.generate(rendered_chats, sampling_params=temp_sampling_params)
            return self._extract_generation_text(outputs, flag_regroup, n)

        normal_chats, empty_prefill_chats = self._split_empty_assistant_prefills(chats)

        chat = getattr(self.model, "chat", None)
        if chat is not None:
            outputs = self._generate_split_chat_batches(
                chat,
                normal_chats,
                empty_prefill_chats,
                temp_sampling_params,
                add_generation_prompt,
                continue_final_message,
            )
        else:
            outputs = self._generate_split_rendered_chat_batches(
                normal_chats,
                empty_prefill_chats,
                temp_sampling_params,
                add_generation_prompt,
                continue_final_message,
            )

        return self._extract_generation_text(outputs, flag_regroup, n)

    def _can_use_chat_template(self):
        return self._uses_mistral_format() or getattr(self.tokenizer, "chat_template", None) is not None

    @staticmethod
    def _render_chats_without_template(chats, continue_final_message=True):
        rendered_chats = []

        for chat in chats:
            parts = []
            for message_index, message in enumerate(chat):
                content = str(message.get("content", ""))
                is_final_message = message_index == len(chat) - 1
                if (
                    continue_final_message
                    and is_final_message
                    and message.get("role") == "assistant"
                ):
                    parts.append(content)
                else:
                    parts.append(content.rstrip() + "\n")
            rendered_chats.append("".join(parts))

        return rendered_chats

    def _split_empty_assistant_prefills(self, chats):
        normal_chats = []
        empty_prefill_chats = []

        for index, chat in enumerate(chats):
            if self._has_empty_final_assistant(chat):
                empty_prefill_chats.append((index, chat[:-1]))
            else:
                normal_chats.append((index, chat))

        return normal_chats, empty_prefill_chats

    @staticmethod
    def _has_empty_final_assistant(chat):
        if not chat:
            return False

        final_message = chat[-1]
        if not isinstance(final_message, Mapping):
            return False

        return (
            final_message.get("role") == "assistant"
            and str(final_message.get("content", "")).strip() == ""
        )

    def _generate_split_chat_batches(self, chat_fn, normal_chats, empty_prefill_chats, sampling_params, add_generation_prompt, continue_final_message):
        outputs_by_index = {}

        if normal_chats:
            indexes, batch = zip(*normal_chats)
            outputs = chat_fn(
                list(batch),
                sampling_params=sampling_params,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
            outputs_by_index.update(zip(indexes, outputs))

        if empty_prefill_chats:
            indexes, batch = zip(*empty_prefill_chats)
            outputs = chat_fn(
                list(batch),
                sampling_params=sampling_params,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            outputs_by_index.update(zip(indexes, outputs))

        return [outputs_by_index[index] for index in sorted(outputs_by_index)]

    def _generate_split_rendered_chat_batches(self, normal_chats, empty_prefill_chats, sampling_params, add_generation_prompt, continue_final_message):
        outputs_by_index = {}

        if normal_chats:
            indexes, batch = zip(*normal_chats)
            rendered_chats = self.tokenizer.apply_chat_template(
                list(batch),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
            outputs = self.model.generate(rendered_chats, sampling_params=sampling_params)
            outputs_by_index.update(zip(indexes, outputs))

        if empty_prefill_chats:
            indexes, batch = zip(*empty_prefill_chats)
            rendered_chats = self.tokenizer.apply_chat_template(
                list(batch),
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            outputs = self.model.generate(rendered_chats, sampling_params=sampling_params)
            outputs_by_index.update(zip(indexes, outputs))

        return [outputs_by_index[index] for index in sorted(outputs_by_index)]

    @classmethod
    def _normalize_chat_token_ids(cls, tokenized_chats):
        if isinstance(tokenized_chats, Mapping):
            tokenized_chats = tokenized_chats["input_ids"]

        tokenized_chats = cls._to_python_list(tokenized_chats)
        if not tokenized_chats:
            return []

        if isinstance(tokenized_chats[0], Integral):
            return [[int(token_id) for token_id in tokenized_chats]]

        normalized_chats = []
        for token_ids in tokenized_chats:
            if isinstance(token_ids, Mapping):
                token_ids = token_ids["input_ids"]

            token_ids = cls._to_python_list(token_ids)
            if not all(isinstance(token_id, Integral) for token_id in token_ids):
                raise TypeError(f"Expected integer chat token ids, got {type(token_ids[0]).__name__}.")

            normalized_chats.append([int(token_id) for token_id in token_ids])

        return normalized_chats
        
    @staticmethod
    def _to_python_list(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    
    # TO implement
    def reset_statistic(self):
        self.num_pmt_toks, self.num_gen_toks, self.num_tot_prmt, self.num_tot_gens = 0, 0, 0, 0

    # TO implement
    def get_statistic(self):
        return {"num_pmt_toks" : int(self.num_pmt_toks), "num_gen_toks" : int(self.num_gen_toks), "num_tot_prmt" : int(self.num_tot_prmt), "num_tot_gens" : int(self.num_tot_gens)}
    
    # TO implement
    def get_num_gen_tokens(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        batch_encode_plus = getattr(self.tokenizer, "batch_encode_plus", None)
        if batch_encode_plus is not None:
            try:
                encodings = batch_encode_plus(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
                return [len(ids) for ids in encodings["input_ids"]]
            except AttributeError:
                pass

        if callable(self.tokenizer):
            try:
                encodings = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
                input_ids = self._extract_token_ids(encodings)
                if input_ids and isinstance(input_ids[0], (list, tuple)):
                    return [len(ids) for ids in input_ids]
                return [len(input_ids)]
            except (AttributeError, TypeError, ValueError):
                pass

        return [len(self._encode_text(text)) for text in texts]

    def _encode_text(self, text):
        encode = getattr(self.tokenizer, "encode", None)
        if encode is not None:
            try:
                return self._extract_token_ids(encode(text, add_special_tokens=False))
            except TypeError:
                return self._extract_token_ids(encode(text))

        encode_plus = getattr(self.tokenizer, "_encode_plus", None)
        if encode_plus is not None:
            try:
                return self._extract_token_ids(encode_plus(text, add_special_tokens=False))
            except TypeError:
                return self._extract_token_ids(encode_plus(text))

        raise AttributeError(f"{self.tokenizer.__class__.__name__} cannot encode generated text.")

    @classmethod
    def _extract_token_ids(cls, encoded):
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif hasattr(encoded, "ids"):
            encoded = encoded.ids

        return cls._to_python_list(encoded)

    def _load_model(self):
        tokenizer_path = self.tokenizer_path if self.tokenizer_path else self.model_path
        use_mistral_format = self._uses_mistral_format()

        kvargs = {}
        if self._quantized in self.kvc_allowed_quant:
            print(f"Using model with quantized kv_cache_dtype: {self._quantized}")
            kvargs = {
                "kv_cache_dtype": self._quantized,
                "calculate_kv_scales": True,
            }

        llm_kwargs = {
            "model": self.model_path,
            "tokenizer": tokenizer_path,
            "max_num_seqs": self._batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "seed": self._seed,
            "enforce_eager": self._enforce_eager,
            "dtype": self._dtype,
            "trust_remote_code": self._trust_remote_code,
            "disable_custom_all_reduce": self._disable_custom_all_reduce,
            **kvargs,
        }

        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = self._max_model_len

        if self._max_cudagraph_capture_size is not None:
            llm_kwargs["max_cudagraph_capture_size"] = self._max_cudagraph_capture_size

        if self._language_model_only:
            llm_kwargs["language_model_only"] = True

        if self._attention_backend:
            llm_kwargs["attention_backend"] = self._attention_backend

        if self._tokenizer_mode is not None:
            llm_kwargs["tokenizer_mode"] = self._tokenizer_mode
        elif use_mistral_format:
            llm_kwargs["tokenizer_mode"] = "mistral"

        if self._config_format is not None:
            llm_kwargs["config_format"] = self._config_format
        elif use_mistral_format:
            llm_kwargs["config_format"] = "mistral"

        if self._load_format is not None:
            llm_kwargs["load_format"] = self._load_format
        elif use_mistral_format:
            llm_kwargs["load_format"] = "mistral"

        if VLLM_VERSION <= Version("0.7.2"):
            llm_kwargs["device"] = self._device
        else:
            llm_kwargs["logits_processors"] = [WrappedPerReqLogitsRetriever]

        return self.AUTO_MODEL_CLASS(**llm_kwargs)

    def _load_tokenizer(self):
        tokenizer_path = self.tokenizer_path if self.tokenizer_path else self.model_path
        tokenizer_kwargs = {
            "trust_remote_code": self._trust_remote_code,
            "padding_side": self._padding_side,
        }
        tokenizer_path_lower = str(tokenizer_path).lower()
        if "mistral" in tokenizer_path_lower or "ministral" in tokenizer_path_lower:
            tokenizer_kwargs["fix_mistral_regex"] = True

        return self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    def _uses_mistral_format(self):
        model_text = f"{self.model_path} {self.tokenizer_path or ''}".lower()
        return "ministral-3" in model_text or "mistral-3" in model_text or "pixtral" in model_text

    def _set_sampling_params(self, generation_args):
        if len(generation_args) == 0:
            return SamplingParams()
        else:
            return SamplingParams(**generation_args)
    
    def _loglikelihood(self, prompts):
        sp = SamplingParams(prompt_logprobs=0, temperature=1, max_tokens=1)
        with torch.no_grad():
            model_output = self.model.generate(prompts, sampling_params=sp)
        
        log_likelihoods = []
        perplexities = []
        for request in model_output:
            prompt = request.prompt_logprobs[1:]
            list_logprops = [next(iter(logprobs.values())).logprob for logprobs in prompt]
            sum_logprops = np.sum(list_logprops)
            log_likelihoods.append(sum_logprops)
            n_toks = len(list_logprops)
            perplexities.append(np.exp(-sum_logprops/n_toks))

            # statistics:
            self.num_pmt_toks += np.sum(n_toks)
            self.num_gen_toks += 1 # here the model generates only one token per prompt.
            self.num_tot_prmt += 1
            self.num_tot_gens += 1

        return log_likelihoods, perplexities

    def _get_temp_updated_sampling_params(self, params):
        new_params = self._generation_args.copy()
        new_params.update(params)
        return SamplingParams(**new_params)
