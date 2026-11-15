"""UTF-8 токенизатор без обучаемого словаря и внешних зависимостей."""

from collections.abc import Iterable
from typing import ClassVar


class ByteTokenizer:
    """Каждый байт UTF-8 становится токеном; три ID служебные."""

    vocab_size: ClassVar[int] = 259
    eos_id: ClassVar[int] = 256
    bos_id: ClassVar[int] = 257
    pad_id: ClassVar[int] = 258
    name: ClassVar[str] = "utf8-bytes-v1"

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        # Один русский символ занимает несколько байтов. Неизвестных слов нет:
        # Любой Unicode-текст кодируется тем же фиксированным набором.
        tokens = list(text.encode("utf-8"))

        if add_bos:
            tokens.insert(0, self.bos_id)

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        result = bytearray()

        for token in tokens:
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"Unknown byte token: {token}")

            if token < 256:
                result.append(token)

        # Генерация может остановиться посреди многобайтного символа.
        # Замена сохраняет готовую часть текста без ошибки при выводе ответа.
        return result.decode("utf-8", errors="replace")
