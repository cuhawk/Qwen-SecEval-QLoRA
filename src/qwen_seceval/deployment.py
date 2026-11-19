"""Generate deployment arguments without importing any serving runtime."""
from pathlib import Path
from typing import Literal

from pydantic import Field

from .schema import StrictModel, Text


class ServeConfig(StrictModel):
    model: Text
    served_model_name: Text = "qwen-seceval"
    host: Text = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_model_len: int = Field(default=4096, gt=0)
    tokenizer: Text | None = None

    def as_vllm_config(self) -> dict:
        return {
            key.replace("_", "-"): value
            for key, value in self.model_dump(exclude_none=True).items()
        }


GGUFType = Literal["Q4_K_M", "Q5_K_M", "Q8_0"]
GGUF_TYPES = ("Q4_K_M", "Q5_K_M", "Q8_0")


def gguf_plan(model: Path, intermediate: Path, output: Path,
              quantization: GGUFType, llama_cpp: Path) -> dict:
    if quantization not in GGUF_TYPES:
        raise ValueError(f"unsupported GGUF quantization {quantization!r}")
    if intermediate.resolve() == output.resolve():
        raise ValueError("intermediate and output must differ")
    return {
        "commands": [
            ["python", str(llama_cpp / "convert_hf_to_gguf.py"), str(model),
             "--outfile", str(intermediate), "--outtype", "f16"],
            [str(llama_cpp / "build/bin/llama-quantize"), str(intermediate),
             str(output), quantization],
        ],
        "output": str(output),
    }
