"""Проверка русской справки в Windows pipes с западной кодировкой."""

import os
import subprocess
import sys

import pytest


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
