# Qwen-SecEval-QLoRA

Production 4-bit NF4 **QLoRA + DPO** fine-tuning and deterministic structured JSON evaluation pipeline for security incident & UI extraction workloads on **Qwen2.5 / Qwen2-VL**, with GGUF/vLLM serving manifests and 95% Wilson CI scoring.

## Offline quick start

Python 3.12 or newer is required. Install the package and run the offline suite.

```sh
python -m pip install -e '.[test]'
python -m pytest -q
qwen-seceval build work/train --count 100 --seed 42
qwen-seceval validate sft work/train/sft.jsonl
qwen-seceval validate dpo work/train/dpo.jsonl
```

The builder writes chat-format `sft.jsonl`, conversational `dpo.jsonl`,
`references.jsonl`, and PNGs under `images/`. AppSec cases cover defensive
control triage. UI cases ask for field extraction from locally rendered,
fictional record screens. Preferred responses conform to the finding schema,
and rejected responses contain a plausible severity or field error. Image
paths are relative to the JSONL file. Generate equivalent fixtures with the
same seed, count, and Pillow version. Use separate held-out data for measuring
generalization. The generated references describe the cases in the same
directory and are intended for pipeline checks and fixture scoring.

## Training and deployment

```sh
qwen-seceval train examples/train_vision.yaml --check
python -m pip install -e '.[train]'
qwen-seceval train examples/train_sft.yaml
qwen-seceval train examples/train_dpo.yaml
qwen-seceval merge examples/train_dpo.yaml work/dpo-adapter work/merged
```

Run training on a CUDA host with access to the model weights. The runtime
uses Transformers, bitsandbytes, PEFT and TRL, with 4-bit NF4 double
quantization and completion-only SFT. DPO can start from a fresh LoRA
adapter or use the SFT adapter as the initial policy and frozen reference.
The `modality` setting selects AppSec text rows or UI image rows. To run
vision DPO, copy the vision example, use the DPO JSONL and `stage: dpo`,
set a new output directory, and point `sft_adapter` at the vision SFT
output. Use a model `revision` commit for reproducible runs.
Paths in training YAML resolve from the current working directory.

The trainer runs without Unsloth. Model downloads, trainer dependencies,
adapter merging, conversion and serving are outside the offline suite.

```sh
qwen-seceval gguf work/merged work/model-f16.gguf work/model-q4.gguf \
  work/quantize.json --llama-cpp /path/to/llama.cpp
qwen-seceval serve-config work/merged work/vllm.yaml
vllm serve --config work/vllm.yaml
```

The GGUF command writes an argument-array manifest. Pass `--run` on a host
with a built llama.cpp and the converter dependencies to convert the merged
text model and quantize it. For OpenAI-compatible screenshot inference, use
vLLM with the merged Hugging Face vision model. The generated config binds to
localhost by default and exposes `/v1/chat/completions`.

## Evaluation

Save raw model responses as JSONL records with an `id` and an `output`
string. Pass the corresponding reference file and choose output paths.

```sh
qwen-seceval evaluate work/train/references.jsonl work/predictions.jsonl \
  --json work/report.json --markdown work/report.md
```

The strict schema pass and severity exact-match rates use all reference IDs
as the denominator. Missing outputs, malformed JSON, duplicate object keys,
extra schema fields and task mismatches fail both rate metrics. Duplicate
record IDs and unknown prediction IDs are input errors. Field micro-F1 uses
exact key/value pairs. A wrong value contributes one false positive and one
false negative, and an invalid or missing response contributes false
negatives for all expected fields. F1 is undefined when there are no
expected or predicted field pairs.

JSON `metrics` values follow AgentReliabilityLab's `MetricValue` shape,
with sample size `n`, `ci_low`, `ci_high`, and
`source: deterministic`. Wilson intervals apply to the two binomial rates.
F1 has null interval bounds. Undefined metrics are omitted from JSON and
shown as N/A in Markdown. The top-level counts expose the scoring totals.
See [the synthetic sample report](examples/sample_eval_report.json).

## Verified Google Colab GPU Benchmark (Base vs. 4-bit NF4 QLoRA + DPO)

Executed cleanly on **NVIDIA L4 (22.03 GB VRAM, `torch.bfloat16`)** via Google Colab Pro+ (`25` held-out incident-triage evaluation cases; telemetry stored in [`benchmarks/colab_qlora_dpo_metrics.json`](benchmarks/colab_qlora_dpo_metrics.json)).

| Metric | Value |
| :--- | :--- |
| **Base Model & Quantization** | `Qwen/Qwen2.5-1.5B-Instruct` (`4-bit NF4 double-quant (bitsandbytes)`) |
| **LoRA Adapter Config** | `r=16`, `alpha=32`, `18,464,768` trainable params (`392` tensors, `73.9 MB` `.safetensors`) |
| **SFT Loss (45 steps)** | `2.8643` -> `0.0197` |
| **DPO Alignment (15 steps)** | Frozen LoRA-disabled reference log-probs (`train_seconds = 35.97s`, peak VRAM `2005 MB`) |
| **Base Strict-JSON Pass Rate** | `0/25` (`0.0%`, markdown-wrapped prose) @ `17.25 tok/s` |
| **Post-QLoRA+DPO Strict-JSON Pass Rate** | **`25/25` (`100.0%`, raw minified JSON)** @ `10 tok/s` |
| **Severity Exact Match** | `0/25` (`0.0%`) -> **`10/25` (`40.0%`)** |
| **Adapter SHA-256** | `c1cdb8772a17191f3434c4f8f5b9de1f3d874494f0dda0a5371999ed4103d7e8` |
