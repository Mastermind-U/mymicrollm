"""Команды для данных, обучения, генерации и подсчёта параметров."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from microllm.model import ModelConfig


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/model"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)

    integers = (
        "steps",
        "batch_size",
        "sequence_length",
        "warmup_steps",
        "accumulation_steps",
        "eval_every",
        "eval_batches",
        "save_every",
        "seed",
    )
    floats = (
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "grad_clip",
    )

    for field in integers:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int)

    for field in floats:
        parser.add_argument(f"--{field.replace('_', '-')}", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM на NumPy: все производные и AdamW записаны вручную"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Число потоков BLAS (по умолчанию 4; задаётся до импорта NumPy)",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Скачать открытый корпус")
    download.add_argument(
        "dataset", choices=("tinystories", "tinyshakespeare")
    )
    download.add_argument("--output-dir", type=Path)

    prepare = commands.add_parser(
        "prepare", help="Преобразовать UTF-8 в токены"
    )
    prepare.add_argument("dataset", choices=("tinystories", "tinyshakespeare"))
    prepare.add_argument("--raw-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path)

    info = commands.add_parser(
        "info", help="Посчитать параметры без выделения RAM"
    )
    info.add_argument("--config", type=Path, default=Path("configs/250m.json"))

    train_parser = commands.add_parser(
        "train", help="Запустить ручной backprop"
    )
    _add_training_arguments(train_parser)

    generate_parser = commands.add_parser("generate", help="Продолжить текст")
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", default="Once upon a time")
    generate_parser.add_argument("--max-new-tokens", type=int, default=200)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=40)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--context-length", type=int)

    return parser


def _load_model_config(path: Path) -> ModelConfig:
    from microllm.model import ModelConfig

    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)

    if not isinstance(values, dict):
        raise ValueError("Model config must be a JSON object")

    return ModelConfig(**values)


def _run_training(args: argparse.Namespace) -> None:
    from microllm.checkpoint import read_checkpoint_metadata
    from microllm.model import ModelConfig, Transformer
    from microllm.training import TrainConfig, train

    if args.resume is not None:
        metadata = read_checkpoint_metadata(args.resume)
        config = ModelConfig(**metadata["config"])
        values = dict(metadata["training_state"]["train_config"])

        if args.config is not None:
            requested = _load_model_config(args.config)

            if dataclasses.asdict(requested) != dataclasses.asdict(config):
                raise ValueError("--config differs from resume checkpoint")
    else:
        config = _load_model_config(args.config or Path("configs/250m.json"))
        values = dataclasses.asdict(TrainConfig())
        values["sequence_length"] = min(128, config.context_length)

    for field in dataclasses.fields(TrainConfig):
        value = getattr(args, field.name)

        if value is not None:
            values[field.name] = value

    training_config = TrainConfig(**values)
    model = Transformer(config)

    print(f"Parameters: {model.parameter_count():,}", flush=True)

    checkpoint = train(
        model,
        args.data_dir,
        args.output_dir,
        training_config,
        resume=args.resume,
        stop_after=args.stop_after,
    )

    print(f"Checkpoint: {checkpoint}")


def _execute(args: argparse.Namespace) -> None:
    if args.command == "download":
        from microllm.data import download_dataset

        output = args.output_dir or Path("data/raw") / args.dataset

        print(download_dataset(args.dataset, output))
    elif args.command == "prepare":
        from microllm.data import prepare_dataset

        raw = args.raw_dir or Path("data/raw") / args.dataset
        output = args.output_dir or Path("data/tokenized") / args.dataset

        print(prepare_dataset(args.dataset, raw, output))
    elif args.command == "info":
        from microllm.model import estimate_parameter_count

        config = _load_model_config(args.config)
        count = estimate_parameter_count(config)

        print(
            json.dumps(
                {
                    "config": dataclasses.asdict(config),
                    "parameters": count,
                    "weights_float32_gib": count * 4 / 1024**3,
                    "weights_gradients_adam_gib": count * 16 / 1024**3,
                },
                indent=2,
            )
        )
    elif args.command == "train":
        _run_training(args)
    elif args.command == "generate":
        from microllm.checkpoint import load_checkpoint
        from microllm.inference import generate

        loaded = load_checkpoint(args.checkpoint)

        # Learned positions за пределами обучающей длины ещё не обучались.
        trained_length = loaded.training_state.get("train_config", {}).get(
            "sequence_length", loaded.model.config.context_length
        )
        window = (
            min(trained_length, loaded.model.config.context_length)
            if args.context_length is None
            else args.context_length
        )

        print(
            generate(
                loaded.model,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
                context_length=window,
            )
        )


def main() -> None:
    # Pipes в Windows могут выбрать cp1252. Помощь и UTF-8 текст модели
    # должны выводиться одинаково в терминал и при перенаправлении в файл.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")

    # У NumPy нет обязательного threadpoolctl: BLAS читает env при импорте.
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ[variable] = str(args.threads)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        _execute(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
