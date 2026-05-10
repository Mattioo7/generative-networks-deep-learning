from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm as auto_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from tqdm.std import tqdm as terminal_tqdm

from .config import ProgressBackend

T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    *,
    enabled: bool,
    backend: ProgressBackend,
    description: str,
    leave: bool = False,
):
    if not enabled:
        return iterable

    factory = {
        "auto": auto_tqdm,
        "notebook": notebook_tqdm,
        "terminal": terminal_tqdm,
    }[backend]

    return factory(
        iterable,
        desc=description,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.5,
    )


def stage(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)
