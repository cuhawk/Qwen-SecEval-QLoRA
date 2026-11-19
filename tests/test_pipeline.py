import json
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from qwen_seceval.cli import main
from qwen_seceval.data import build_dataset, image_path, read_rows, write_rows
from qwen_seceval.deployment import gguf_plan
from qwen_seceval.evaluation import evaluate, markdown_summary, wilson_interval
from qwen_seceval.schema import DPORow, EvalRow, Finding, Prediction, SFTRow, parse_finding
from qwen_seceval.training import TrainConfig, training_examples


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(root, count=4, seed=10)
    return root


def test_reproducible(tmp_path, dataset):
    copy = tmp_path / "copy"
    build_dataset(copy, count=4, seed=10)
    files = sorted(path.relative_to(dataset) for path in dataset.rglob("*") if path.is_file())
    assert len(files) == 5
    assert all((dataset / path).read_bytes() == (copy / path).read_bytes() for path in files)


def test_sft_format(dataset):
    rows = list(read_rows(dataset / "sft.jsonl", SFTRow))
    assert [row.task for row in rows] == ["appsec", "ui"] * 2
    assert all([m.role for m in row.messages] == ["system", "user", "assistant"] for row in rows)


def test_dpo_matches_sft(dataset):
    sft = list(read_rows(dataset / "sft.jsonl", SFTRow))
    dpo = list(read_rows(dataset / "dpo.jsonl", DPORow))
    for supervised, preference in zip(sft, dpo, strict=True):
        assert supervised.id == preference.id
        assert supervised.messages == preference.prompt + preference.chosen
        assert preference.chosen != preference.rejected


def test_screenshots_are_images(dataset):
    for row in read_rows(dataset / "sft.jsonl", SFTRow):
        if row.image:
            with Image.open(image_path(dataset / "sft.jsonl", row.image)) as image:
                assert (image.format, image.size) == ("PNG", (480, 240))
                assert len(image.getcolors(maxcolors=5000)) > 2
                assert parse_finding(row.messages[-1].content).fields["record"] not in row.messages[1].content


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_count(tmp_path, count):
    with pytest.raises(ValueError, match="positive"):
        build_dataset(tmp_path, count=count)


@pytest.mark.parametrize("value", [
    {"severity": "HIGH"}, {"severity": 2}, {"task": "unknown"},
    {"fields": {"a": 1}}, {"fields": []}, {"fields": {"": "valid"}},
    {"title": ""}, {"title": "  "}, {"extra": "x"}, {"evidence": None},
])
def test_invalid_finding(value, dataset):
    reference = next(read_rows(dataset / "references.jsonl", EvalRow)).expected
    with pytest.raises(ValueError):
        Finding.model_validate(reference.model_dump() | value)


@pytest.mark.parametrize("raw", [
    "{}", "[]", "null", "NaN", '{"a": Infinity}', '{"a":{"b":0,"b":1}}',
    '"""json\n{}\n"""', "{} trailing",
])
def test_invalid_wire_json(raw):
    with pytest.raises(ValueError):
        parse_finding(raw)


def test_preference_cannot_repeat_chosen(dataset):
    first = next(read_rows(dataset / "dpo.jsonl", DPORow)).model_dump()
    with pytest.raises(ValueError, match="semantically"):
        DPORow.model_validate(first | {"rejected": first["chosen"]})


def test_message_order_required(dataset):
    first = next(read_rows(dataset / "sft.jsonl", SFTRow)).model_dump()
    with pytest.raises(ValueError, match="system message"):
        SFTRow.model_validate(first | {"messages": first["messages"][::-1]})


def test_ui_requires_image(dataset):
    rows = list(read_rows(dataset / "sft.jsonl", SFTRow))
    with pytest.raises(ValueError, match="image"):
        SFTRow.model_validate(rows[1].model_dump() | {"image": None})


def test_duplicate_row_ids(dataset, tmp_path):
    reference = next(read_rows(dataset / "references.jsonl", EvalRow))
    path = tmp_path / "duplicates.jsonl"
    write_rows(path, [reference, reference])
    with pytest.raises(ValueError, match=r"duplicates.jsonl:2: duplicate id"):
        list(read_rows(path, EvalRow))


@pytest.mark.parametrize("image", ["../outside.png", "/outside.png", "missing.png"])
def test_invalid_image_path(dataset, image):
    with pytest.raises(ValueError):
        image_path(dataset / "sft.jsonl", image)


@pytest.mark.parametrize(("k", "n", "expected"), [
    (0, 1, (0, 0.79345069)), (1, 1, (0.20654931, 1)),
    (5, 10, (0.23659309, 0.76340691)), (8, 10, (0.49016247, 0.94331785)),
    (0, 10, (0, 0.27753280)), (10, 10, (0.72246720, 1)),
])
def test_wilson_known_values(k, n, expected):
    assert wilson_interval(k, n) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(("k", "n"), [(1, 0), (-1, 10), (0, -1), (True, 1), (1, 1.0)])
def test_invalid_wilson_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_empty_evaluation(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    report = evaluate(path, path)
    assert wilson_interval(0, 0) is None
    assert report["metrics"] == {}
    assert markdown_summary(report).count("N/A") == 3


def test_mixed_evaluation(dataset, tmp_path):
    references = dataset / "references.jsonl"
    first, second, third, _ = list(read_rows(references, EvalRow))
    prediction_path = tmp_path / "predictions.jsonl"
    wrong = second.expected.model_copy(update={"severity": "low", "fields": {"status": "Unknown"}})
    write_rows(prediction_path, [
        Prediction(id=third.id, output="invalid"),
        Prediction(id=second.id, output=wrong.model_dump_json()),
        Prediction(id=first.id, output=first.expected.model_dump_json()),
    ])
    report = evaluate(references, prediction_path)
    metrics = report["metrics"]
    assert report["missing_predictions"] == 1
    assert metrics["strict_json_schema_pass_rate"]["value"] == 0.5
    assert metrics["severity_exact_match"]["value"] == 0.25
    assert metrics["field_micro_f1"]["value"] == pytest.approx(6 / 14)
    assert metrics["field_micro_f1"]["ci_low"] is None
    for item in metrics.values():
        assert set(item) == {"name", "value", "unit", "n", "ci_low", "ci_high", "source"}
        assert item["n"] == 4


def test_unknown_predictions(dataset, tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    write_rows(prediction_path, [Prediction(id="unknown", output="")])
    with pytest.raises(ValueError, match="unknown prediction IDs"):
        evaluate(dataset / "references.jsonl", prediction_path)


@pytest.mark.parametrize("stage", ["sft", "dpo"])
@pytest.mark.parametrize("modality", ["vision", "text"])
def test_offline_training_examples(dataset, stage, modality):
    config = TrainConfig(model="qwen", dataset=str(dataset / f"{stage}.jsonl"),
                         output_dir="out", stage=stage, modality=modality)
    examples = training_examples(config)
    assert len(examples) == 2
    assert ("images" in examples[0]) == (modality == "vision")
    assert examples[0]["prompt"][0]["role"] == "system"
    assert ("completion" in examples[0]) == (stage == "sft")


def test_vision_truncation_rejected():
    with pytest.raises(ValueError, match="preserve image tokens"):
        TrainConfig(model="qwen", dataset="data", output_dir="out", modality="vision", max_length=512)


def test_cli_help_without_gpu_runtime():
    result = subprocess.run([sys.executable, "-m", "qwen_seceval", "--help"],
                            capture_output=True, text=True, check=True)
    assert "serve-config" in result.stdout


def test_cli_build_validate(tmp_path, capsys):
    root = tmp_path / "generated"
    assert main(["build", str(root), "--count", "2"]) == 0
    assert main(["validate", "sft", str(root / "sft.jsonl")]) == 0
    assert "Validated 2 rows" in capsys.readouterr().out
    (root / "images/synthetic-000001.png").unlink()
    assert main(["validate", "sft", str(root / "sft.jsonl")]) == 1
    assert "missing image" in capsys.readouterr().err


def test_cli_report(dataset, tmp_path):
    references = dataset / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    report = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    write_rows(predictions, [
        Prediction(id=row.id, output=row.expected.model_dump_json())
        for row in read_rows(references, EvalRow)
    ])
    assert main(["evaluate", str(references), str(predictions),
                 "--json", str(report), "--markdown", str(markdown)]) == 0
    assert json.loads(report.read_text())["metrics"]["severity_exact_match"]["value"] == 1
    assert "95% Wilson CI" in markdown.read_text()


def test_cli_serving(tmp_path):
    config = tmp_path / "vllm.yaml"
    assert main(["serve-config", "model.gguf", str(config), "--tokenizer", "original"]) == 0
    assert yaml.safe_load(config.read_text()) == {
        "model": "model.gguf", "tokenizer": "original", "served-model-name": "qwen-seceval",
        "host": "127.0.0.1", "port": 8000, "max-model-len": 4096,
    }
    assert main(["serve-config", "model", str(config), "--port", "0"]) == 1


def test_cli_gguf_manifest(tmp_path):
    manifest = tmp_path / "plan.json"
    args = ["gguf", "merged", "f16.gguf", "q4.gguf", str(manifest), "--llama-cpp", "llama.cpp"]
    assert main(args) == 0
    commands = json.loads(manifest.read_text())["commands"]
    assert commands[0][-2:] == ["--outtype", "f16"]
    assert commands[1][-1] == "Q4_K_M"


def test_gguf_rejects_output_collision(tmp_path):
    output = tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="must differ"):
        gguf_plan(tmp_path, output, output, "Q4_K_M", tmp_path)
