from itertools import groupby
from typing import Any, Callable, List, Sequence

KEY_FUNCTION = Callable[[Any], Any]


def sort(keys: Sequence, values: Sequence, *, key_fn: KEY_FUNCTION) -> List[Any]:
    """
    Sort values sequence by keys sequence.
    """
    value_map = {key_fn(v): v for v in values}
    return list(map(lambda k: value_map.get(k), keys))


def group_sort(keys: Sequence, values: Sequence, *, key_fn: KEY_FUNCTION) -> List[Any]:
    """
    Sort and group value sequence by key sequence.

    Often used in one to many and many to many cases.
    """
    values = sorted(values, key=key_fn)
    value_map = {key: list(data) for key, data in groupby(values, key=key_fn)}
    return list(map(lambda k: value_map.get(k), keys))
