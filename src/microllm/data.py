"""Потоковая загрузка открытых текстов и запись uint16 токенов на диск."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from array import array
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import structlog

from microllm.tokenizer import ByteTokenizer

LOGGER = structlog.get_logger(__name__)
CHUNK_SIZE = 1024 * 1024
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
SHAKESPEARE_REVISION = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"


@dataclass(frozen=True)
class SourceFile:
    filename: str
    url: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetSource:
    name: str
    title: str
    homepage: str
    revision: str
    license: str
    attribution: str
    files: tuple[SourceFile, ...]


DATASETS: dict[str, DatasetSource] = {
    "tinystories": DatasetSource(
        name="tinystories",
        title="TinyStories",
        homepage="https://huggingface.co/datasets/roneneldan/TinyStories",
        revision=TINYSTORIES_REVISION,
        license="CDLA-Sharing-1.0 (declared in upstream dataset card)",
        attribution="Ronen Eldan and Yuanzhi Li; synthetic English stories",
        files=(
            SourceFile(
                filename="TinyStories-train.txt",
                url=(
                    "https://huggingface.co/datasets/roneneldan/TinyStories/"
                    f"resolve/{TINYSTORIES_REVISION}/TinyStories-train.txt"
                ),
                size=1924281556,
                sha256=(
                    "c5cf5e22ff13614e830afbe61a99fbcbe"
                    "8bcb7dd72252b989fa1117a368d401f"
                ),
            ),
            SourceFile(
                filename="TinyStories-valid.txt",
                url=(
                    "https://huggingface.co/datasets/roneneldan/TinyStories/"
                    f"resolve/{TINYSTORIES_REVISION}/TinyStories-valid.txt"
                ),
                size=19447282,
                sha256=(
                    "94e431816c4cce81ff71e4408ff8d3bda"
                    "9a42e8d2663986697c3954288cb38b4"
                ),
            ),
            SourceFile(
                filename="SOURCE_README.md",
                url=(
                    "https://huggingface.co/datasets/roneneldan/TinyStories/"
                    f"resolve/{TINYSTORIES_REVISION}/README.md"
                ),
            ),
        ),
    ),
    "tinyshakespeare": DatasetSource(
        name="tinyshakespeare",
        title="Tiny Shakespeare",
        homepage="https://github.com/karpathy/char-rnn",
        revision=SHAKESPEARE_REVISION,
        license=(
            "MIT (upstream repository declaration); "
            "original Shakespeare works are public domain"
        ),
        attribution=(
            "William Shakespeare; corpus distributed by Andrej Karpathy"
        ),
        files=(
            SourceFile(
                filename="input.txt",
                url=(
                    "https://raw.githubusercontent.com/karpathy/char-rnn/"
                    f"{SHAKESPEARE_REVISION}/data/tinyshakespeare/input.txt"
                ),
                size=1115394,
                sha256=(
                    "86c4e6aa9db7c042ec79f339dcb96d42"
                    "b0075e16b8fc2e86bf0ca57e2dc565ed"
                ),
            ),
            SourceFile(
                filename="SOURCE_README.md",
                url=(
                    "https://raw.githubusercontent.com/karpathy/char-rnn/"
                    f"{SHAKESPEARE_REVISION}/Readme.md"
                ),
            ),
        ),
    ),
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_source(path: Path, source: SourceFile) -> str:
    if source.size is not None and path.stat().st_size != source.size:
        raise ValueError(f"Unexpected size: {path}")

    digest = file_sha256(path)

    if source.sha256 is not None and digest != source.sha256:
        raise ValueError(f"SHA256 mismatch: {path}")

    return digest


def download_file(source: SourceFile, destination: Path) -> str:
    """Продолжить .part при HTTP Range и публиковать после проверки SHA256."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return verify_source(destination, source)

    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0

    if source.size is not None and offset == source.size:
        digest = verify_source(temporary, source)
        temporary.replace(destination)

        return digest

    if source.size is not None and offset > source.size:
        raise ValueError(f"Partial file exceeds expected size: {temporary}")

    # Источники используют HTTPS с обычной проверкой сертификата Python.
    if not source.url.startswith("https://"):
        raise ValueError("Dataset downloads require HTTPS")

    request = urllib.request.Request(  # noqa: S310
        source.url,
        headers={
            "User-Agent": "microllm/0.1 (open research dataset download)"
        },
    )

    if offset:
        request.add_header("Range", f"bytes={offset}-")

    LOGGER.info(
        "download_started", filename=source.filename, offset_bytes=offset
    )

    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        resumed = response.status == 206

        if resumed and not response.headers.get(
            "Content-Range", ""
        ).startswith(
            f"bytes {offset}-",
        ):
            raise ValueError("Server returned an unexpected Content-Range")

        # Когда сервер проигнорировал Range, начинаем заново, иначе получится
        # незаметная склейка начала файла с уже скачанным префиксом.
        mode = "ab" if resumed else "wb"
        received = offset if resumed else 0
        reported = received

        with temporary.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                received += len(chunk)

                if received - reported >= 64 * CHUNK_SIZE:
                    LOGGER.info(
                        "download_progress",
                        filename=source.filename,
                        received_bytes=received,
                    )
                    reported = received

    digest = verify_source(temporary, source)
    temporary.replace(destination)
    LOGGER.info(
        "download_verified",
        filename=source.filename,
        size_bytes=destination.stat().st_size,
    )

    return digest


def download_dataset(name: str, output_dir: Path) -> Path:
    source = DATASETS[name]
    files: list[dict[str, object]] = []

    for item in source.files:
        destination = output_dir / item.filename
        digest = download_file(item, destination)
        files.append(
            {
                **asdict(item),
                "size": destination.stat().st_size,
                "sha256": digest,
            },
        )

    manifest = {**asdict(source), "files": files}
    path = output_dir / "manifest.json"
    write_json(path, manifest)

    return path


def iter_stories(path: Path) -> Iterator[str]:
    """Держать в памяти один документ, разделённый маркером TinyStories."""
    parts: list[str] = []

    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            fragments = line.split("<|endoftext|>")
            parts.append(fragments[0])

            for fragment in fragments[1:]:
                story = "".join(parts).strip()

                if story:
                    yield story

                parts = [fragment]

    story = "".join(parts).strip()

    if story:
        yield story


def write_document(stream: BinaryIO, text: str) -> int:
    tokenizer = ByteTokenizer()
    # array пишет компактные числа, а не Python int по 28+ байтов на токен.
    # Байтовый порядок фиксирован, поэтому файл переносим между платформами.
    tokens = array("H", tokenizer.encode(text, add_bos=True, add_eos=True))

    if sys.byteorder != "little":
        tokens.byteswap()

    tokens.tofile(stream)

    return len(tokens)


def write_tokens(documents: Iterator[str], path: Path) -> dict[str, object]:
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    document_count = 0

    with temporary.open("wb") as stream:
        for document in documents:
            count += write_document(stream, document)
            document_count += 1

    if count == 0:
        temporary.unlink()
        raise ValueError(f"No documents found for {path}")

    temporary.replace(path)

    return {
        "file": path.name,
        "tokens": count,
        "documents": document_count,
        "sha256": file_sha256(path),
    }


def prepare_dataset(name: str, raw_dir: Path, output_dir: Path) -> Path:
    """Создать train.bin/val.bin без смешивания официальных выборок."""
    source = DATASETS[name]
    output_dir.mkdir(parents=True, exist_ok=True)

    if name == "tinystories":
        train_documents = iter_stories(raw_dir / "TinyStories-train.txt")
        val_documents = iter_stories(raw_dir / "TinyStories-valid.txt")
        split_description = "Official upstream train and validation files"
    else:
        # Маленький Shakespeare (около 1 МБ) допустимо прочитать целиком.
        # Граница проходит между параграфами; ни один документ не делится.
        text = (raw_dir / "input.txt").read_text(encoding="utf-8")
        paragraphs = [part for part in text.split("\n\n") if part.strip()]

        if len(paragraphs) < 2:
            raise ValueError(
                "At least two paragraphs are needed for splitting"
            )

        boundary = min(max(int(len(paragraphs) * 0.9), 1), len(paragraphs) - 1)
        train_documents = iter(paragraphs[:boundary])
        val_documents = iter(paragraphs[boundary:])
        split_description = (
            "First 90% of paragraphs train, final 10% validation"
        )

    train = write_tokens(train_documents, output_dir / "train.bin")
    validation = write_tokens(val_documents, output_dir / "val.bin")
    metadata: dict[str, object] = {
        "dataset": name,
        "source": source.homepage,
        "revision": source.revision,
        "license": source.license,
        "tokenizer": ByteTokenizer.name,
        "vocab_size": ByteTokenizer.vocab_size,
        "bos_id": ByteTokenizer.bos_id,
        "eos_id": ByteTokenizer.eos_id,
        "pad_id": ByteTokenizer.pad_id,
        "dtype": "uint16",
        "byteorder": "little",
        "split": split_description,
        "train_tokens": train["tokens"],
        "val_tokens": validation["tokens"],
        "train": train,
        "validation": validation,
    }
    manifest = raw_dir / "manifest.json"

    if manifest.exists():
        metadata["source_manifest_sha256"] = file_sha256(manifest)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        metadata["source_files"] = manifest_data["files"]

    path = output_dir / "metadata.json"
    write_json(path, metadata)

    return path
