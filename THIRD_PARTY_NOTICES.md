# Third-party software and models

GPUEnsemble's own code is under the MIT license ([LICENSE](LICENSE)). It contains no code copied from other projects. What it bundles or downloads comes under its own license:

## In the Windows zip

| What | Where from | License |
| --- | --- | --- |
| `GPUEnsemble.exe` (the window app, only in the `-app` zip) | this repository ([app/windows](app/windows)) | MIT, as the rest of GPUEnsemble. It runs on .NET Framework 4.8, which is part of Windows and not in the zip; the reference assemblies used to build it ([Microsoft.NETFramework.ReferenceAssemblies](https://www.nuget.org/packages/Microsoft.NETFramework.ReferenceAssemblies), MIT) are used at build time only |
| Python 3.12 (embeddable package), unmodified | [python.org](https://www.python.org/downloads/windows/) | [PSF License Agreement](https://docs.python.org/3/license.html); the text is in the zip as `python\LICENSE.txt` |

## Downloaded when it runs

Nothing below is in this repository or in the zip. `setup` / `up` download it to the user's PC from the publisher's own site.

| What | Where from | License |
| --- | --- | --- |
| llama.cpp (llama-server, ggml-rpc-server and their libraries), prebuilt | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) releases | MIT |
| CUDA runtime libraries, for the CUDA build of llama.cpp | the same llama.cpp release (`cudart-llama-bin-win-cuda-*.zip`) | NVIDIA's CUDA Toolkit license (redistributable libraries) |
| Models (GGUF files) | [Hugging Face](https://huggingface.co), from the repositories named in [catalog.json](planner/src/gpuensemble_planner/catalog.json) | each model's own license, shown on its page |

Models are not part of GPUEnsemble and are not redistributed by it: each one is downloaded from its Hugging Face page and comes under the license its publisher gives there. Some licenses restrict what a model may be used for, so check the page before use. The models the app and `recommend` choose from ([catalog.json](planner/src/gpuensemble_planner/catalog.json); one page can hold several sizes of the same model):

| Model | Downloaded from |
| --- | --- |
| DeepSeek-V4-Flash 284B-A13B Q4_K_XL | [unsloth/DeepSeek-V4-Flash-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF) |
| GLM-5.3-Flash 313B-A18B Q4_K_XL | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) |
| MiniMax-M3 426B-A23B Q4_K_XL | [unsloth/MiniMax-M3-GGUF](https://huggingface.co/unsloth/MiniMax-M3-GGUF) |
| Qwen3.5-397B-A17B Q4_K_M | [unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) |
| MiMo-V2.6-Flash 309B-A15B MXFP4 | [ggml-org/MiMo-V2.6-Flash-RL-GGUF](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF) |
| Hy3 299B-A21B Q4_K_M | [bartowski/Hy3-GGUF](https://huggingface.co/bartowski/Hy3-GGUF) |
| Qwen3.8-Flash-Next 177B-A6B Q4_K_XL | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| Step-3.7-Flash 197B-A11B Q4_K_XL | [unsloth/Step-3.7-Flash-GGUF](https://huggingface.co/unsloth/Step-3.7-Flash-GGUF) |
| Qwen3-235B-A22B-2507 Q4_K_M | [unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF) |
| gpt-oss-120b MXFP4 | [ggml-org/gpt-oss-120b-GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) |
| Qwen3.8-27B Q4_K_M | [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) |
| Nemotron-3-Ultra 550B-A55B Q2_K_XL | [unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF) |
| Qwen3.5-122B-A10B Q4_K_M | [unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) |
| Mistral-Small-4-119B Q4_K_M | [unsloth/Mistral-Small-4-119B-2603-GGUF](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF) |
| Nemotron-3-Super-120B-A12B Q4_K_M | [unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF) |
| Nemotron-3.5-Super-VL 120B-A12B Q4_K_XL | [unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) |
| Gemma-4-31B Q4_K_M | [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) |
| Qwen3.6-35B-A3B Q4_K_M | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| GLM-4.5-Air Q4_K_M | [ggml-org/GLM-4.5-Air-GGUF](https://huggingface.co/ggml-org/GLM-4.5-Air-GGUF) |
| Llama-4-Scout 109B-A17B Q4_K_M | [ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF) |
| Mistral-Medium-3.5 125B Q4_K_M | [unsloth/Mistral-Medium-3.5-128B-GGUF](https://huggingface.co/unsloth/Mistral-Medium-3.5-128B-GGUF) |
| Devstral-2 123B Q4_K_M | [unsloth/Devstral-2-123B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Devstral-2-123B-Instruct-2512-GGUF) |
| Qwen3-Coder-Next 80B-A3B Q4_K_M | [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) |
| Gemma-4-26B-A4B Q4_K_M | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |
| Nemotron-3.5-Bolt-74B-A9B Q4_K_XL | [unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF) |
| Olmo-3.1-32B-Think Q4_K_M | [bartowski/allenai_Olmo-3.1-32B-Think-GGUF](https://huggingface.co/bartowski/allenai_Olmo-3.1-32B-Think-GGUF) |
| Nemotron-3.5-Lightning-30B-A3B Q4_0 | [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) |
| Granite-4.2-30B Q4_K_M | [bartowski/granite-4.2-30b-GGUF](https://huggingface.co/bartowski/granite-4.2-30b-GGUF) |
| Qwen3-32B Q4_K_M | [Qwen/Qwen3-32B-GGUF](https://huggingface.co/Qwen/Qwen3-32B-GGUF) |
| gpt-oss-20b MXFP4 | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) |
| Devstral-Small-2-24B Q4_K_M | [unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF) |
| Gemma-4-12B Q4_K_M | [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) |
| Ministral-3-14B Q4_K_M | [unsloth/Ministral-3-14B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF) |
| Phi-4 14B Q4_K_M | [unsloth/phi-4-GGUF](https://huggingface.co/unsloth/phi-4-GGUF) |
| Qwen3.5-9B Q4_K_M | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Granite-4.2-8B Q4_K_M | [bartowski/granite-4.2-8b-GGUF](https://huggingface.co/bartowski/granite-4.2-8b-GGUF) |
| Ministral-3-8B Q4_K_M | [unsloth/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF) |
| Qwen3-8B Q4_K_M | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| Gemma-4-E4B Q4_0 | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |
| Qwen3.5-4B Q4_K_M | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |

`catalog --update` (run by the app at start) may add newer GGUF models it finds on Hugging Face; they are marked as new in the list, and their licenses are on their own pages too.

## Ideas

The layer split and the MoE expert offload follow the ideas of [Strata](https://github.com/Niko1221/Strata) (MIT). No code was taken from it.
