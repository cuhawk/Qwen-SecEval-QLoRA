"""Command-line entry point for the offline pipeline and optional runtime."""
import argparse
from pathlib import Path
import subprocess
import sys

from PIL import Image
import yaml

from .data import build_dataset, image_path, read_rows
from .deployment import GGUF_TYPES, ServeConfig, gguf_plan
from .evaluation import evaluate, markdown_summary, write_json
from .schema import DPORow, EvalRow, Prediction, SFTRow
from .training import load_train_config, merge_adapter, train, training_examples

SCHEMAS = {"sft": SFTRow, "dpo": DPORow, "references": EvalRow, "predictions": Prediction}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="qwen-seceval")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="generate synthetic JSONL and PNG screenshots")
    build.add_argument("directory", type=Path)
    build.add_argument("--count", type=int, default=24)
    build.add_argument("--seed", type=int, default=42)

    validate = commands.add_parser("validate", help="validate JSONL and any referenced images")
    validate.add_argument("kind", choices=SCHEMAS)
    validate.add_argument("dataset", type=Path)

    score = commands.add_parser("evaluate", help="export a benchmark report")
    score.add_argument("references", type=Path)
    score.add_argument("predictions", type=Path)
    score.add_argument("--json", dest="json_path", type=Path, required=True)
    score.add_argument("--markdown", type=Path)

    training = commands.add_parser("train", help="run an SFT or DPO job")
    training.add_argument("config", type=Path)
    training.add_argument("--check", action="store_true", help="validate inputs offline")

    merge = commands.add_parser("merge", help="merge a trained PEFT adapter")
    merge.add_argument("config", type=Path)
    merge.add_argument("adapter", type=Path)
    merge.add_argument("output", type=Path)

    serve = commands.add_parser("serve-config", help="generate vLLM OpenAI server configuration")
    serve.add_argument("model")
    serve.add_argument("output", type=Path)
    serve.add_argument("--name", default="qwen-seceval")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--max-model-len", default=4096, type=int)
    serve.add_argument("--tokenizer")

    gguf = commands.add_parser("gguf", help="plan or run llama.cpp GGUF conversion and quantization")
    gguf.add_argument("model", type=Path)
    gguf.add_argument("intermediate", type=Path)
    gguf.add_argument("output", type=Path)
    gguf.add_argument("manifest", type=Path)
    gguf.add_argument("--llama-cpp", required=True, type=Path)
    gguf.add_argument("--quantization", choices=GGUF_TYPES, default="Q4_K_M")
    gguf.add_argument("--run", action="store_true")
    return root


def run(args: argparse.Namespace) -> None:
    match args.command:
        case "build":
            build_dataset(args.directory, args.count, args.seed)
        case "validate":
            count = 0
            for row in read_rows(args.dataset, SCHEMAS[args.kind]):
                if isinstance(row, (SFTRow, DPORow)) and row.image:
                    with Image.open(image_path(args.dataset, row.image)) as image:
                        image.verify()
                count += 1
            print(f"Validated {count} rows")
        case "evaluate":
            if args.markdown and args.markdown.resolve() == args.json_path.resolve():
                raise ValueError("JSON and Markdown outputs must differ")
            report = evaluate(args.references, args.predictions)
            write_json(args.json_path, report)
            if args.markdown:
                args.markdown.write_text(markdown_summary(report), encoding="utf-8")
        case "train":
            config = load_train_config(args.config)
            if args.check:
                print(f"Validated {len(training_examples(config))} training examples")
            else:
                train(config)
        case "merge":
            merge_adapter(load_train_config(args.config), args.adapter, args.output)
        case "serve-config":
            config = ServeConfig(
                model=args.model, served_model_name=args.name, host=args.host, port=args.port,
                max_model_len=args.max_model_len, tokenizer=args.tokenizer,
            )
            args.output.write_text(yaml.safe_dump(config.as_vllm_config()), encoding="utf-8")
        case "gguf":
            if args.manifest.resolve() in {args.intermediate.resolve(), args.output.resolve()}:
                raise ValueError("manifest and GGUF paths must differ")
            plan = gguf_plan(args.model, args.intermediate, args.output, args.quantization, args.llama_cpp)
            write_json(args.manifest, plan)
            if args.run:
                for command in plan["commands"]:
                    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print(f"qwen-seceval: {exc}", file=sys.stderr)
        return 1
    return 0
