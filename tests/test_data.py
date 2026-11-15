"""Проверки Unicode, границ документов и воспроизводимости выборок."""

import hashlib
import io
import json
import sys
import urllib.request
from array import array
from pathlib import Path

import pytest

from microllm.data import (
    SourceFile,
    download_file,
    iter_stories,
    prepare_dataset,
    write_tokens,
)
from microllm.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    original = "Привет, мир! 🐈 café\n"
    tokens = tokenizer.encode(original, add_bos=True, add_eos=True)

    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id
    assert tokenizer.decode(tokens) == original

    with pytest.raises(ValueError, match="Unknown byte token"):
        tokenizer.decode([259])


def test_documents_and_binary_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "stories.txt"
    source.write_text(
        "one\n<|endoftext|>\n\nдва<|endoftext|>tail",
        encoding="utf-8",
    )
    assert list(iter_stories(source)) == ["one", "два", "tail"]
    binary = tmp_path / "train.bin"
    result = write_tokens(iter_stories(source), binary)
    expected = [
        257,
        111,
        110,
        101,
        256,
        257,
        208,
        180,
        208,
        178,
        208,
        176,
        256,
    ]
    expected += [257, 116, 97, 105, 108, 256]
    tokens = array("H")
    tokens.frombytes(binary.read_bytes())

    if sys.byteorder != "little":
        tokens.byteswap()

    assert list(tokens) == expected
    assert result["tokens"] == len(expected)
    assert result["documents"] == 3


def test_official_splits_remain_separate(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "TinyStories-train.txt").write_text("TRAIN", encoding="utf-8")
    (raw / "TinyStories-valid.txt").write_text("VALID", encoding="utf-8")
    output = tmp_path / "tokens"
    meta_path = prepare_dataset("tinystories", raw, output)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert metadata["train_tokens"] == 7
    assert metadata["val_tokens"] == 7
    assert b"T\x00R\x00A\x00I\x00N\x00" in (output / "train.bin").read_bytes()
    assert b"V\x00A\x00L\x00I\x00D\x00" in (output / "val.bin").read_bytes()
    assert metadata["vocab_size"] == 259


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "train.bin"

    with pytest.raises(ValueError, match="No documents"):
        write_tokens(iter(()), destination)

    assert not destination.exists()


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        status: int,
        headers: dict[str, str],
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers


@pytest.mark.parametrize("supports_range", [True, False])
def test_download_resume_or_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supports_range: bool,
) -> None:
    payload = b"complete file content"
    source = SourceFile(
        filename="file.txt",
        url="https://example.test/file.txt",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    destination = tmp_path / source.filename
    partial = tmp_path / "file.txt.part"
    partial.write_bytes(payload[:3])

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: int,
    ) -> FakeResponse:
        assert request.get_header("Range") == "bytes=3-"
        assert timeout == 60

        if supports_range:
            content_range = f"bytes 3-{len(payload) - 1}/{len(payload)}"

            return FakeResponse(
                payload[3:],
                206,
                {"Content-Range": content_range},
            )

        return FakeResponse(payload, 200, {})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert download_file(source, destination) == source.sha256
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_download_checksum_mismatch_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(
        request: urllib.request.Request,
        timeout: int,
    ) -> FakeResponse:
        assert request.full_url == "https://example.test/file.txt"
        assert timeout == 60

        return FakeResponse(b"broken", 200, {})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    source = SourceFile(
        "file.txt",
        "https://example.test/file.txt",
        sha256="0" * 64,
    )
    destination = tmp_path / source.filename

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_file(source, destination)

    assert not destination.exists()
    assert (tmp_path / "file.txt.part").read_bytes() == b"broken"
