"""Проверка русской справки в Windows pipes с западной кодировкой."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_training import make_data


@pytest.mark.parametrize("arguments", [["--help"], ["train", "--help"]])
def test_help_uses_utf8_even_with_legacy_console_encoding(
    arguments: list[str],
) -> None:
    # Команда фиксирована; вход пользователя не передаётся shell.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "microllm", *arguments],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )

    assert "usage:" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("log_format", ["console", "json"])
def test_training_logs_and_metrics(
    tmp_path: Path,
    log_format: str,
) -> None:
    data = make_data(tmp_path / "data")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "context_length": 8,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "d_ff": 16,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "модель"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "microllm",
            "--log-format",
            log_format,
            "train",
            "--config",
            str(config),
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
            "--steps",
            "1",
            "--sequence-length",
            "4",
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert not result.stdout
    metrics = json.loads((output / "metrics.jsonl").read_text("utf-8"))
    if log_format == "console":
        assert "\x1b[" in result.stderr
        assert "training_step" in result.stderr
        assert re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", result.stderr)
        assert metrics["step"] == 1
        return

    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [event["event"] for event in events] == [
        "model_initialized",
        "training_started",
        "training_step",
        "checkpoint_saved",
        "training_finished",
    ]
    assert all(event["level"] == "info" for event in events)
    assert events[1]["output_dir"] == str(output)
    assert {key: events[2][key] for key in metrics} == metrics
