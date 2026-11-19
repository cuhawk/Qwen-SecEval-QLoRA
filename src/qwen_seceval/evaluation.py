"""Deterministic scoring and versioned benchmark summaries."""
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

from .data import read_rows
from .schema import EvalRow, Prediction, parse_finding

Z_95 = NormalDist().inv_cdf(0.975)


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    """Return the two-sided 95% Wilson interval, or None for an empty sample."""
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("expected integer counts with 0 <= successes <= total")
    if total == 0:
        return None
    rate = successes / total
    adjustment = Z_95**2 / total
    center = (rate + adjustment / 2) / (1 + adjustment)
    margin = Z_95 * sqrt((rate * (1 - rate) + adjustment / 4) / total) / (1 + adjustment)
    return (0.0 if successes == 0 else center - margin,
            1.0 if successes == total else center + margin)


def metric(name: str, value: float, total: int,
           interval: tuple[float, float] | None = None) -> dict:
    return {
        "name": name, "value": value, "unit": "ratio", "n": total,
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "source": "deterministic",
    }


@dataclass
class FieldCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def update(self, expected: dict[str, str], actual: dict[str, str]) -> None:
        self.true_positives += len(expected.items() & actual.items())
        self.false_positives += len(actual.items() - expected.items())
        self.false_negatives += len(expected.items() - actual.items())

    def summary(self) -> dict:
        twice_matched = 2 * self.true_positives
        denominator = twice_matched + self.false_positives + self.false_negatives
        return {
            "value": twice_matched / denominator if denominator else None,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def evaluate(references: Path, predictions: Path) -> dict:
    expected = {row.id: row.expected for row in read_rows(references, EvalRow)}
    outputs = {row.id: row.output for row in read_rows(predictions, Prediction)}
    unknown = outputs.keys() - expected.keys()
    if unknown:
        raise ValueError(f"unknown prediction IDs: {', '.join(sorted(unknown))}")
    valid = severity = missing = 0
    fields = FieldCounts()
    for row_id, reference in expected.items():
        if row_id not in outputs:
            missing += 1
            fields.update(reference.fields, {})
            continue
        try:
            predicted = parse_finding(outputs[row_id])
            if predicted.task != reference.task:
                raise ValueError("prediction task differs from reference")
        except ValueError:
            fields.update(reference.fields, {})
            continue
        valid += 1
        severity += predicted.severity == reference.severity
        fields.update(reference.fields, predicted.fields)
    metrics = {}
    total = len(expected)
    counts = fields.summary()
    if total:
        for name, successes in (
            ("strict_json_schema_pass_rate", valid), ("severity_exact_match", severity)
        ):
            metrics[name] = metric(name, successes / total, total, wilson_interval(successes, total))
    if counts["value"] is not None:
        name = "field_micro_f1"
        metrics[name] = metric(name, counts["value"], total)
    return {
        "schema_version": "1.0",
        "benchmark": "qwen-seceval",
        "n_examples": len(expected),
        "missing_predictions": missing,
        "metrics": metrics,
        "counts": {
            "schema_passes": valid, "severity_matches": severity,
            "field_true_positives": fields.true_positives,
            "field_false_positives": fields.false_positives,
            "field_false_negatives": fields.false_negatives,
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def markdown_summary(report: dict) -> str:
    lines = [
        "# Qwen SecEval benchmark", "",
        f"Examples **{report['n_examples']}** · Missing predictions **{report['missing_predictions']}**",
        "", "| Metric | Score | 95% Wilson CI |", "| --- | ---: | --- |",
    ]
    for name in ("strict_json_schema_pass_rate", "severity_exact_match", "field_micro_f1"):
        result = report["metrics"].get(name, {})
        lower, upper = result.get("ci_low"), result.get("ci_high")
        bounds = "—" if lower is None else f"{_percent(lower)} to {_percent(upper)}"
        lines.append(f"| {name.replace('_', ' ')} | {_percent(result.get('value'))} | {bounds} |")
    return "\n".join(lines) + "\n"
