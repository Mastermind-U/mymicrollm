"""Сохранение чисел и состояния обучения без pickle и исполняемого кода."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from microllm.model import ModelConfig, Transformer

type FloatArray = NDArray[np.float32]


@dataclass
class LoadedCheckpoint:
    model: Transformer
    optimizer_arrays: dict[str, FloatArray]
    training_state: dict[str, Any]


def read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))

    if (
        not isinstance(metadata, dict)
        or metadata.get("format_version") != 1
        or not isinstance(metadata.get("config"), dict)
        or not isinstance(metadata.get("training_state"), dict)
    ):
        raise ValueError("Invalid or unsupported checkpoint metadata")

    return metadata


def _write_arrays(
    archive: zipfile.ZipFile,
    arrays: dict[str, FloatArray],
    prefix: str,
) -> dict[str, str]:
    entries: dict[str, str] = {}

    for index, (name, array) in enumerate(sorted(arrays.items())):
        entry = f"{prefix}/{index:05d}.npy"
        entries[name] = entry

        # Потоковая запись не создаёт ещё одну копию всех весов в RAM.
        with archive.open(entry, "w", force_zip64=True) as output:
            np.lib.format.write_array(output, array, allow_pickle=False)

    return entries


def save_checkpoint(
    path: Path,
    model: Transformer,
    *,
    optimizer_arrays: dict[str, FloatArray] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            metadata = {
                "format_version": 1,
                "config": asdict(model.config),
                "parameters": _write_arrays(
                    archive, model.parameters, "parameters"
                ),
                "optimizer": _write_arrays(
                    archive, optimizer_arrays or {}, "optimizer"
                ),
                "training_state": training_state or {},
            }

            archive.writestr(
                "metadata.json",
                json.dumps(metadata, ensure_ascii=False, allow_nan=False),
            )

        # При прерывании предыдущий checkpoint остаётся целым.
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_entries(value: object, prefix: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Invalid checkpoint array index")

    entries: dict[str, str] = {}

    for index, (name, entry) in enumerate(sorted(value.items())):
        if (
            not isinstance(name, str)
            or not isinstance(entry, str)
            or entry != f"{prefix}/{index:05d}.npy"
        ):
            raise ValueError("Invalid checkpoint array entry")

        entries[name] = entry

    return entries


def _read_array(archive: zipfile.ZipFile, entry: str) -> FloatArray:
    with archive.open(entry) as source:
        array = np.lib.format.read_array(source, allow_pickle=False)

    if array.dtype != np.float32 or not np.all(np.isfinite(array)):
        raise ValueError(f"Checkpoint array {entry} must be finite float32")

    return array


def load_checkpoint(
    path: Path,
    *,
    load_optimizer: bool = False,
    model: Transformer | None = None,
) -> LoadedCheckpoint:
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))

        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")

        config = ModelConfig(**metadata["config"])

        if model is None:
            model = Transformer(config)
        elif model.config != config:
            raise ValueError("Checkpoint architecture does not match model")

        parameters = _array_entries(metadata["parameters"], "parameters")
        optimizer = _array_entries(metadata["optimizer"], "optimizer")
        expected_entries = {"metadata.json", *parameters.values()}
        expected_entries.update(optimizer.values())
        actual_entries = archive.namelist()

        if (
            len(actual_entries) != len(set(actual_entries))
            or set(actual_entries) != expected_entries
            or set(parameters) != set(model.parameters)
        ):
            raise ValueError("Checkpoint contains missing or extra arrays")

        for name, entry in parameters.items():
            array = _read_array(archive, entry)

            if array.shape != model.parameters[name].shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")

            np.copyto(model.parameters[name], array)

        optimizer_arrays: dict[str, FloatArray] = {}

        if load_optimizer:
            for name, entry in optimizer.items():
                parameter_name = name.removeprefix("m/").removeprefix("v/")

                if (
                    name not in {f"m/{parameter_name}", f"v/{parameter_name}"}
                    or parameter_name not in model.parameters
                ):
                    raise ValueError("Invalid optimizer parameter name")

                array = _read_array(archive, entry)

                if array.shape != model.parameters[parameter_name].shape:
                    raise ValueError(f"Optimizer shape mismatch: {name}")

                optimizer_arrays[name] = array

        state = metadata["training_state"]

        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint training state")

    return LoadedCheckpoint(model, optimizer_arrays, state)
