from itertools import groupby
from typing import Any, Callable, List, Optional, Sequence, TypeVar

ValueT = TypeVar("ValueT")
KeyT = TypeVar("KeyT")

KEY_FUNCTION = Callable[[ValueT], KeyT]


def sort(
    keys: Sequence[KeyT],
    values: Sequence[ValueT],
    *,
    key_fn: KEY_FUNCTION[ValueT, KeyT]
) -> List[Optional[ValueT]]:
    """
    Sort values sequence by keys sequence.
    """
    value_map = {key_fn(v): v for v in values}
    return list(map(lambda k: value_map.get(k), keys))  # type: ignore


def group_sort(
    keys: Sequence[KeyT],
    values: Sequence[ValueT],
    *,
    key_fn: KEY_FUNCTION[ValueT, KeyT]
) -> List[Optional[List[ValueT]]]:
    """
    Sort and group value sequence by key sequence.

    Often used in one to many and many to many cases.
    """
    values = sorted(values, key=key_fn)  # type: ignore
    value_map = {key: list(data) for key, data in groupby(values, key=key_fn)}
    return list(map(lambda k: value_map.get(k), keys))
