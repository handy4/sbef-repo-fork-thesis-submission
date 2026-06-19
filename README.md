<p align="center">
  <img src="https://github.com/insait-institute/quantization-affects-social-bias/blob/master/docs/static/images/example_gh.svg" width="100%">
</p>

# How Quantization Shapes Bias in LLMs

> #### Federico Marcuzzi, Xuefei Ning, Roy Schwartz, and Iryna Gurevych
>

This repository includes the code and scripts to reproduce the experiments presented in the EACL 2026 paper [How Quantization Shapes Bias in Large Language Models](https://arxiv.org/abs/2508.18088) (paper [website](https://insait-institute.github.io/quantization-affects-social-bias/)). The code can also be used to test social bias on large language models compatible with the HuggingFace library. Our framework is built on top of [COMPL-AI](https://github.com/compl-ai/compl-ai).

## Abstract
This work presents a comprehensive evaluation of how quantization affects model bias, with particular attention to its impact on individual demographic subgroups.
We focus on weight and activation quantization strategies and examine their effects across a broad range of bias types, including stereotypes, fairness, toxicity, and sentiment.
We employ both probability- and generated text-based metrics across 13 benchmarks and evaluate models that differ in architecture family and reasoning ability.
Our findings show that quantization has a nuanced impact on bias: while it can reduce model toxicity and does not significantly impact sentiment, it tends to slightly increase stereotypes and unfairness in generative tasks, especially under aggressive compression.
These trends are generally consistent across demographic categories and subgroups, and model types, although their magnitude depends on the specific setting.
Overall, our results highlight the importance of carefully balancing efficiency and ethical considerations when applying quantization in practice.

---

## Setup

Clone the repository and fetch all submodules:

```bash
git clone https://github.com/insait-institute/quantization-affects-social-bias.git
cd quantization-affects-social-bias
git submodule update --init --recursive
```

[Note] When a path is required to run a script, please provide the absolute path to avoid errors.

[Recommended] Set the HuggingFace home to the model folder at the root of the repository, and export your HF token, which is required to download the benchmarks and models.
```bash
export HF_HOME="./models"
export HF_TOKEN="..."
```

Create the two Conda environments needed to run the Social Bias Evaluation Framework and the quantization library:

* To set up the framework environment, use the ```framework_env.yaml``` file:

  ```bash
  conda env create -f framework_env.yaml
  ```

* To make the framework work with recent models, update the framework using the ```framework_env_vllm_0.17.1.yaml``` file and manually update vllm to v0.21.0 and Transformers to v5.8.1:

  ```bash
  conda activate bias_eval
  conda env update -f framework_env_vllm_0.17.1.yaml 
  python -m pip install --upgrade "vllm==0.21.0" "transformers==5.8.1"
  ```

* [Optional] To set up the quantization library environment, use the ```compression_env.yaml``` file:

  ```bash
  conda env create -f compression/compression_env.yaml
  ```

Download the necessary datasets to run the evaluation:

```bash
conda activate bias_eval
python helper_tools/download_datasets.py
```

[Optional] Download the un-quantized pre-trained models into the `MODELS_DIR` folder:

```bash
conda activate bias_eval
bash helper_tools/download_models.py <MODELS_DIR>
```

[Optional] Quantize the model as described in the article. After quantization, each model will be saved in a dedicated folder within `MODELS_DIR` (note: `MODELS_DIR` is the folder containing the root folder of the models to be quantized).

```bash
conda activate compress
cd compress_models <MODELS_DIR>
bash compress_models.sh
```

## Test Framework

* [Fast] To test the installation of the Social Bias Evaluation Framework on a dummy model, run the following command:

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_test.sh
  ```

* [Slow] To test the framework on the LLM model saved in the `MODEL_PATH` folder, run the following script. The script will load the LLM and run the evaluation on a subset of each evaluation benchmark.

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_debug.sh <MODEL_PATH>
  ```

## Run Full Evaluation

* To fully evaluate a model, use the following commands, where `MODEL_PATH` is the model folder and `CONFIG_PATH` is the model configuration file stored in `./configs/models/`:

```bash
  conda activate bias_eval
  cd run_scripts
  bash run.sh <MODEL_PATH> <CONFIG_PATH>
  ```

* To reproduce the evaluation performed in the article, run the following:

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_all.sh <MODELS_DIR>
  ```

## Run LLM-as-a-judge Evaluation

* To run the LLM-as-a-judge evaluation on toxic continuations, use the following, where `BENCH_RESULTS_DIR` is the folder containing the benchmark results (e.g., `results/runs/Qwen2.5-14B-Instruct/1984-04-30_00:00:00`), `BENCH_NAME` can be `bold` or `dt_toxic`, `MODEL_NAME` is the name of the model whose results you want to evaluate, and [optional] `JUDGE_PATH` is the path to the judge model.

```bash
  conda activate bias_eval
  cd run_scripts
  bash run_judge.sh <BENCH_RESULTS_DIR> <BENCH_NAME> <MODEL_NAME> <JUDGE_PATH>
  ```

* To reproduce the  LLM-as-a-judge evaluation performed in the article, run the following:

 ```bash
  conda activate bias_eval
  cd run_scripts
  bash run_judge_all.sh
  ```

## Compute Model Size

To compute the size of the un-quantized model as well as the non-fake-quantized models reported in the article, run the following commands:

 ```bash
conda activate bias_eval
python helper_tools/compute_model_size.py 
```

---

## Citation
```
@inproceedings{DBLP:conf/eacl/MarcuzziNSG26,
  author       = {Federico Marcuzzi and Xuefei Ning and Roy Schwartz and Iryna Gurevych},
  editor       = {Vera Demberg and Kentaro Inui and Llu{\'{\i}}s Marquez},
  title        = {How Quantization Shapes Bias in Large Language Models},
  booktitle    = {Proceedings of the 19th Conference of the European Chapter of the
                  Association for Computational Linguistics, {EACL} 2026 - Volume 1:
                  Long Papers, Rabat, Morocco, March 24-29, 2026},
  pages        = {363--404},
  publisher    = {Association for Computational Linguistics},
  year         = {2026},
  url          = {https://aclanthology.org/2026.eacl-long.17/},
  timestamp    = {Mon, 30 Mar 2026 17:02:29 +0200},
  biburl       = {https://dblp.org/rec/conf/eacl/MarcuzziNSG26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

# Code Changes Summary

## Main Goal

The original framework was written against older model and vLLM behavior. After updating `vllm` and `transformers` for newer models, several modern instruction models exposed compatibility issues:

- Mistral/Ministral/Pixtral models needed Mistral-specific vLLM loading/tokenizer options.
- Some vLLM defaults caused memory pressure, CUDA graph stalls, or backend-specific failures on the Slurm system.
- Some chat-template models generated empty text when benchmarks used raw string prompts.
- Some tokenizer objects exposed by newer vLLM versions did not support the older token-counting assumptions.

## vLLM Model Wrapper

Changed files:

- `src/models/base/vllm_model.py`

### Added Runtime Configuration Pass-Through

The wrapper now reads additional model config fields and passes them to `vllm.LLM(...)`:

- `gpu_memory_utilization`
- `max_model_len`
- `max_cudagraph_capture_size`
- `enforce_eager`
- `language_model_only`
- `tensor_parallel_size`
- `disable_custom_all_reduce`
- `attention_backend`
- `tokenizer_mode`
- `config_format`
- `load_format`
- `use_chat_template_for_generate`

Reason:

Newer vLLM releases expose important compatibility and stability controls through `LLM(...)` arguments. The original wrapper did not expose these controls, which made it hard to work around memory pressure, CUDA graph capture issues, multimodal architecture loading, attention backend problems, and tensor-parallel behavior on the Slurm machine.

### Safer Tensor Parallel Handling

The wrapper now chooses `tensor_parallel_size` from config when provided, and clamps it to the number of visible GPUs.

Reason:

The original behavior used all visible GPUs automatically. That was inconvenient for debugging, could conflict with Slurm allocation details, and could trigger custom all-reduce or multi-GPU behavior that was not needed for smaller/debug runs.

### Mistral/Ministral/Pixtral vLLM Format Handling

For model/tokenizer paths matching `ministral-3`, `mistral-3`, or `pixtral`, the wrapper now defaults these vLLM options:

- `tokenizer_mode="mistral"`
- `config_format="mistral"`
- `load_format="mistral"`

The tokenizer loader also enables `fix_mistral_regex=True` for Mistral/Ministral tokenizers.

Reason:

The newer Mistral-family models are not always loadable with vLLM's generic Hugging Face path. They need Mistral-aware config/tokenizer/loading behavior to avoid architecture, tokenizer, or malformed-regex failures.

### Language-Model-Only Loading

The wrapper can pass `language_model_only=True` to vLLM.

Reason:

Some models, especially Pixtral-like architectures, resolve as conditional-generation or multimodal architectures even when the benchmark only needs text generation. Loading only the language model avoids unnecessary multimodal paths and related compatibility issues.

### More Robust Chat Generation

`generate_chat()` was expanded to support:

- vLLM's native `model.chat(...)` API when available.
- Manual rendering with `tokenizer.apply_chat_template(...)` as a fallback.
- A raw text fallback when no chat template exists.
- Splitting chats with an empty final assistant message from normal chats.
- For empty assistant prefills, forcing `add_generation_prompt=True` and `continue_final_message=False`.

Reason:

Several benchmarks use chat-shaped prompts with assistant prefill text. Newer tokenizer/vLLM combinations can reject empty assistant prefills or handle `continue_final_message` differently. Splitting those cases avoids hangs or empty outputs while preserving intended prefill behavior where it exists.

### EuroLLM Raw Prompt Fix

`generate()` now automatically wraps raw string prompts as a single user chat message for EuroLLM models when the tokenizer has a chat template. This can also be forced or disabled through:

```yaml
use_chat_template_for_generate: true
```

or:

```yaml
use_chat_template_for_generate: false
```

Reason:

EuroLLM-22B-Instruct worked on `bold` because that benchmark used `generate_chat()`, but produced empty generations on `wino_bias`, `bbq`, `dt_fairness`, `discrim_eval_gen`, and similar raw-prompt benchmarks. The likely cause was that EuroLLM expects ChatML/chat-template formatting; raw prompts caused immediate EOS. Wrapping raw prompts through the chat template makes those benchmarks use the format expected by the instruct model.

### Shared Generation Extraction

Generation text extraction and token-statistic updates were centralized in `_extract_generation_text(...)`.

Reason:

Both raw and chat generation paths need the same output normalization and token accounting. Centralizing it reduced duplicated logic and made later fixes apply consistently.

### Token Counting Compatibility

`get_num_gen_tokens()` now handles tokenizers that do not provide `batch_encode_plus`, including vLLM tokenizer backends that expose different callable/encoding APIs.

Reason:

Some newer vLLM tokenizer wrappers are not full Hugging Face tokenizer objects. The original token-counting logic could fail during BOLD/DT toxicity scoring after generation had already succeeded.

## Model Config Schema

Changed files:

- `src/configs/base_model_config.py`

Added Pydantic fields:

- `gpu_memory_utilization`
- `max_model_len`
- `max_cudagraph_capture_size`
- `enforce_eager`
- `language_model_only`
- `tensor_parallel_size`
- `disable_custom_all_reduce`
- `attention_backend`
- `tokenizer_mode`
- `config_format`
- `load_format`
- `use_chat_template_for_generate`

Reason:

The YAML model configs needed to accept the new vLLM runtime controls without validation errors. These fields are consumed by the patched vLLM wrapper.

## Default Model Configs

Changed files:

- `configs/models/default.yaml`
- `configs/models/default_debug.yaml`

The default config was made more conservative:

```yaml
batch_size: 64
gpu_memory_utilization: 0.65
max_model_len: 8192
max_cudagraph_capture_size: 64
enforce_eager: true
language_model_only: true
tensor_parallel_size: 1
disable_custom_all_reduce: true
attention_backend: "TRITON_ATTN"
```

The debug config uses smaller limits:

```yaml
batch_size: 8
max_gen_toks: 512
gpu_memory_utilization: 0.5
max_model_len: 4096
max_cudagraph_capture_size: 16
enforce_eager: true
language_model_only: true
tensor_parallel_size: 1
disable_custom_all_reduce: true
attention_backend: "TRITON_ATTN"
```

Reason:

The original defaults were too optimistic for the updated vLLM/model stack on the Slurm system. The new defaults reduce memory pressure, avoid CUDA graph capture stalls, avoid custom all-reduce issues, and prefer the Triton attention backend after FlashInfer-related failures.

## Environment Variables

If there are issues with newer models or the flashinfer smapler, use the following environment variables:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```


Reason:

The updated vLLM installation attempted to use newer optional kernels/backends that were either unavailable, unstable, or incompatible with the cluster environment. These exports disable problematic DeepGEMM/FlashInfer behavior and force the attention backend used successfully in later runs.
