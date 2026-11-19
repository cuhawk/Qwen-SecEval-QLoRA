"""Build deterministic defensive review and screenshot extraction fixtures."""
import random
from pathlib import Path
from typing import Iterator, TypeVar

from PIL import Image, ImageDraw

from .schema import DPORow, EvalRow, Finding, Message, SFTRow, StrictModel, parse_json

Row = TypeVar("Row", bound=StrictModel)
SYSTEM = (
    "Return one JSON object with task, severity, title, evidence, remediation, and fields. "
    "Use task appsec or ui. Severity must be none, low, medium, high, or critical. "
    "Use a string-to-string object for fields. Report only evidence from the supplied input."
)
CONTROLS = (
    ("Access review overdue", "medium", "access_review", "overdue", "Schedule an access review."),
    ("Backups unverified", "medium", "backup_restore", "unverified", "Verify a backup restoration."),
    ("Alert routing inactive", "high", "alert_routing", "inactive", "Restore and verify alert routing."),
    ("Audit log retention verified", "none", "audit_retention", "verified", "Maintain the retention policy."),
)
STATES = ("Pending", "Approved", "Archived")


def read_rows(path: Path, schema: type[Row]) -> Iterator[Row]:
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = schema.model_validate(parse_json(line))
                if row.id in seen:
                    raise ValueError(f"duplicate id {row.id!r}")
                seen.add(row.id)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield row


def write_rows(path: Path, rows: list[StrictModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")


def screenshot(path: Path, record: str, status: str, theme: int) -> None:
    background, panel, ink = (
        ("#eef2ff", "#ffffff", "#182044"), ("#111827", "#1f2937", "#f9fafb"),
    )[theme]
    image = Image.new("RGB", (480, 240), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 456, 216), radius=16, fill=panel)
    draw.text((48, 47), "RECORD OVERVIEW", fill=ink, font_size=25)
    draw.line((48, 94, 432, 94), fill=ink)
    draw.text((48, 115), f"Record: {record}", fill=ink, font_size=20)
    draw.text((48, 158), f"Status: {status}", fill=ink, font_size=20)
    image.save(path)


def build_dataset(directory: Path, count: int = 24, seed: int = 42) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "images").mkdir(exist_ok=True)
    sft, dpo, references = [], [], []
    for index in range(count):
        row_id = f"synthetic-{index:06d}"
        record = f"R-{rng.randrange(10000, 100000)}-{index}"
        image = None
        if index % 2:
            status = rng.choice(STATES)
            image = f"images/{row_id}.png"
            screenshot(directory / image, record, status, rng.randrange(2))
            prompt = "Extract the record and status from the screenshot."
            finding = Finding(task="ui", severity="none", title="Record overview",
                              evidence=f"{record} is {status}.", remediation="No action required.",
                              fields={"record": record, "status": status})
            alternate = next(value for value in STATES if value != status)
            rejected = finding.model_copy(update={"fields": {"record": record, "status": alternate}})
        else:
            title, severity, control, state, remediation = rng.choice(CONTROLS)
            evidence = f"Review {record} recorded {control} as {state}."
            prompt = f"Triage this synthetic control review. {evidence}"
            finding = Finding(task="appsec", severity=severity, title=title,
                              evidence=evidence, remediation=remediation,
                              fields={"review": record, "control": control, "state": state})
            rejected = finding.model_copy(update={"severity": "high" if severity == "none" else "none"})
        messages = [Message(role="system", content=SYSTEM), Message(role="user", content=prompt)]
        chosen = [Message(role="assistant", content=finding.model_dump_json())]
        negative = [Message(role="assistant", content=rejected.model_dump_json())]
        metadata = dict(id=row_id, task=finding.task, image=image)
        sft.append(SFTRow(**metadata, messages=messages + chosen))
        dpo.append(DPORow(**metadata, prompt=messages, chosen=chosen, rejected=negative))
        references.append(EvalRow(id=row_id, expected=finding))
    write_rows(directory / "sft.jsonl", sft)
    write_rows(directory / "dpo.jsonl", dpo)
    write_rows(directory / "references.jsonl", references)


def image_path(dataset: Path, image: str) -> Path:
    root = dataset.resolve().parent
    relative = Path(image)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root):
        raise ValueError(f"image must be relative to the dataset directory: {image}")
    if not candidate.is_file():
        raise ValueError(f"missing image: {candidate}")
    return candidate
