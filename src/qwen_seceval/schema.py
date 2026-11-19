"""Strict wire schemas shared by training data and the evaluator."""
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Severity = Literal["none", "low", "medium", "high", "critical"]
Task = Literal["appsec", "ui"]
Role = Literal["system", "user", "assistant"]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Finding(StrictModel):
    task: Task
    severity: Severity
    title: Text
    evidence: Text
    remediation: Text
    fields: dict[Text, Text]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def parse_json(raw: str):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def parse_finding(raw: str) -> Finding:
    return Finding.model_validate(parse_json(raw))


class Message(StrictModel):
    role: Role
    content: Text


class DatasetRow(StrictModel):
    id: Text
    task: Task
    image: Text | None = None

    def check_prompt(self, prompt: list[Message]) -> None:
        if [message.role for message in prompt] != ["system", "user"]:
            raise ValueError("expected a system message followed by a user message")
        if (self.task == "ui") != (self.image is not None):
            raise ValueError("only UI rows must have an image")

    def check_completion(self, completion: list[Message]) -> Finding:
        if [message.role for message in completion] != ["assistant"]:
            raise ValueError("expected one assistant completion")
        finding = parse_finding(completion[0].content)
        if finding.task != self.task:
            raise ValueError("completion task differs from row task")
        return finding


class SFTRow(DatasetRow):
    messages: list[Message]

    @model_validator(mode="after")
    def check_messages(self):
        self.check_prompt(self.messages[:-1])
        self.check_completion(self.messages[-1:])
        return self


class DPORow(DatasetRow):
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]

    @model_validator(mode="after")
    def check_pair(self):
        self.check_prompt(self.prompt)
        if self.check_completion(self.chosen) == self.check_completion(self.rejected):
            raise ValueError("chosen and rejected must differ semantically")
        return self


class EvalRow(StrictModel):
    id: Text
    expected: Finding


class Prediction(StrictModel):
    id: Text
    output: str
