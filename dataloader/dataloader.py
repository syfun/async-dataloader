import asyncio
import dataclasses
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from typer import Option

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

KT = TypeVar("KT")
VT = TypeVar("VT")
CT = TypeVar("CT")
Number = Union[int, float]


BatchLoadFn = Callable[[List[KT]], Awaitable[List[VT]]]
BatchScheduleFn = Callable[..., None]
CacheKeyFn = Callable[[KT], CT]


def is_number(n: Any) -> bool:
    return isinstance(n, (int, float))


class CacheMap(Protocol, Generic[KT, VT]):
    def get(self, key: KT) -> Optional[VT]:
        ...

    def set(self, key: KT, value: VT) -> Any:
        ...

    def delete(self, key: KT) -> Any:
        ...

    def clear(self) -> Any:
        ...

    def __contains__(self, key: KT) -> bool:
        ...


class Map(Generic[KT, VT]):
    def __init__(self):
        self._data: Dict[KT, VT] = {}

    def get(self, key: KT) -> Optional[VT]:
        return self._data.get(key)

    def set(self, key: KT, value: VT) -> Any:
        self._data[key] = value
        return None

    def delete(self, key: KT) -> Any:
        return self._data.pop(key, None)

    def clear(self) -> Any:
        self._data = {}
        return None

    def __contains__(self, key: KT) -> bool:
        return key in self._data


@dataclasses.dataclass
class Batch(Generic[KT, VT]):
    has_dispatched: bool
    keys: List[KT]
    futures: List[asyncio.Future]


class Options(Generic[KT, VT, CT]):
    batch: Optional[bool]
    max_batch_size: Optional[Number]
    batch_schedule_fn: Optional[BatchScheduleFn]
    cache: Optional[bool]
    cache_key_fn: Optional[CacheKeyFn]
    cache_map: Optional[CacheMap[KT, Awaitable[VT]]] = None


class DataLoader(Generic[KT, VT, CT]):
    _batch_load_fn: BatchLoadFn
    _max_batch_size: int
    _batch_schedule_fn: BatchScheduleFn
    _cache_key_fn: CacheKeyFn
    _cache_map: Optional[CacheMap[KT, Awaitable[VT]]]
    _batch: Optional[Batch[KT, VT]]
    _loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        batch_load_fn: BatchLoadFn,
        options: Optional[Options[KT, VT, CT]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        if not callable(batch_load_fn):
            raise TypeError(
                f"DataLoader must be constructed with a function which accepts "
                f"Sequence[key] and returns Awaitable[Sequence[value]], but got: {batch_load_fn}."
            )
        self._batch_load_fn = batch_load_fn
        self._max_batch_size = get_valid_max_batch_size(options)
        self._batch_schedule_fn = get_valid_batch_schedule_fn(options)
        self._cache_key_fn = get_valid_cache_key_fn(options)
        self._cache_map = get_valid_cache_map(options)
        self._batch = None
        self._loop = loop

    @property
    def loop(self):
        if not self._loop:
            self._loop = asyncio.get_event_loop()
        return self._loop

    def load(self, key: KT) -> Awaitable[VT]:
        """
        Loads a key, returning a future for the value represented by that key.
        """
        if key is None:
            raise TypeError(
                f"The loader.load() function must be called with a value, "
                f"but got: {key}."
            )
        batch = get_current_batch(self)
        cache_map = self._cache_map
        cache_key = self._cache_key_fn(key)

        # If caching and there is a cache-hit, return cached future.
        if cache_map:
            future = cache_map.get(cache_key)
            if future:
                return future

        # Otherwise, produce a new future for this key, and enqueue it to be
        # dispatched along with the current batch.
        batch.keys.append(cache_key)
        future = self.loop.create_future()
        batch.futures.append(future)

        # If caching, cache this future.
        if cache_map:
            cache_map.set(cache_key, future)
        return future

    def load_many(self, keys: Sequence[KT]) -> Awaitable[Sequence[VT]]:
        """
        Loads multiple keys, return an sequence of futures.
        """
        return asyncio.gather(*[self.load(key) for key in keys])

    def clear(self, key: KT) -> "DataLoader":
        """
        Clears the future at key from the cache, if it exists. Returns itself for
        method chaining.
        """
        if self._cache_map:
            self._cache_map.delete(self._cache_key_fn(key))
        return self

    def clear_all(self) -> "DataLoader":
        """
        Clears the entire cache. To be used when some event results in unknown
        invalidations across this particular Dataloader. Returns itself for
        method chaining.
        """
        if self._cache_map:
            self._cache_map.clear()

        return self

    def prime(self, key: KT, value: Union[VT, Exception]) -> "DataLoader":
        """
        Adds the provided key and value to the cache. If the key already exists,
        no change is made. Returns itself for method chaining.
        To prime the cache with an exception at a key, provide an Exception instance.
        """
        if self._cache_map:
            cache_key = self._cache_key_fn(key)
            # Only add this key if it does not already exist.
            if cache_key not in self._cache_map:
                future = self.loop.create_future()
                if isinstance(value, Exception):
                    future.set_exception(value)
                else:
                    future.set_result(value)
                self._cache_map.set(key, future)

        return self


def get_current_batch(loader: DataLoader[KT, VT, Any]) -> Batch[KT, VT]:
    """
    Either returns the current batch, or creates and schedules a
    dispatch of a new batch for the given loader.
    """
    # If there is an existing batch which has not yet dispatched and is within
    # the limit of the batch size, then return it.
    exiting_batch = loader._batch
    if (
        exiting_batch
        and not exiting_batch.has_dispatched
        and len(exiting_batch.keys) < loader._max_batch_size
    ):
        return exiting_batch

    new_batch = Batch(has_dispatched=False, keys=[], futures=[])
    loader._batch = new_batch
    loader._batch_schedule_fn(
        loader, asyncio.create_task, dispatch_batch(loader, new_batch)
    )
    return new_batch


async def dispatch_batch(loader: DataLoader[KT, VT, Any], batch: Batch[KT, VT]) -> None:
    # Mark this batch as having been dispatched.
    batch.has_dispatched = True

    # If there's nothing to load, return early.
    if len(batch.keys) == 0:
        return None

    # Call the provided batch_load_fn for this loader with the batch's keys.
    batch_coroutine = loader._batch_load_fn(batch.keys)
    if not batch_coroutine or not inspect.isawaitable(batch_coroutine):
        error = TypeError(
            f"DataLoader must be constructed with a function which accepts "
            f"Sequence[key] and returns Awaitable[Sequence[value]], but the function did "
            f"not return a Aawaitable object: {batch_coroutine}."
        )
        return failed_dispatch(loader, batch, error)

    values = await batch_coroutine
    if not isinstance(values, Sequence):
        raise TypeError(
            f"DataLoader must be constructed with a function which accepts "
            f"Sequence[key] and returns Awaitable[Sequence[value]], but the function did "
            f"not return a Awaitable object of Sequence[value]: {values}."
        )

    if len(values) != len(batch.keys):
        raise TypeError(
            "DataLoader must be constructed with a function which accepts "
            "Sequence[key] and returns Awaitable[Sequence[value]], but the function did "
            "not return a Awaitable object of Sequence[value] of the same length as "
            "the Sequence of keys." + f"\n\nKeys: \n {batch.keys}\n\nValues: {values}"
        )

    for future, value in zip(batch.futures, values):
        if isinstance(value, Exception):
            future.set_exception(value)
        else:
            future.set_result(value)


def failed_dispatch(
    loader: DataLoader[KT, VT, Any], batch: Batch[KT, VT], error: Exception
) -> None:
    """
    do not cache individual loads if the entire batch dispatch fails,
    but still set_exception to each future so they do not hang.
    """
    for key, future in zip(batch.keys, batch.futures):
        loader.clear(key)
        future.set_exception(error)


def get_valid_max_batch_size(
    options: Optional[Options[Any, Any, Any]] = None
) -> Number:
    should_batch = not options or options.batch
    if not should_batch:
        return 1

    max_batch_size = options and options.max_batch_size
    if max_batch_size is None:
        return float("inf")

    if not is_number(max_batch_size) or max_batch_size < 1:
        raise TypeError(f"max_batch_size must be a positive number: {max_batch_size}")

    return max_batch_size


def default_batch_schedule_fn(
    loader: DataLoader, callback: Callable, *args: Any
) -> None:
    loader.loop.call_soon(callback, *args)


def get_valid_batch_schedule_fn(
    options: Optional[Options[Any, Any, Any]] = None
) -> BatchScheduleFn:
    batch_schedule_fn = options and options.batch_schedule_fn
    if batch_schedule_fn is None:
        return default_batch_schedule_fn

    if not inspect.isfunction(batch_schedule_fn):
        raise TypeError(f"batch_schedule_fn must be a function: {batch_schedule_fn}")

    return batch_schedule_fn


def default_cache_key_fn(key: Any) -> Any:
    return key


def get_valid_cache_key_fn(
    options: Optional[Options[Any, Any, Any]] = None
) -> CacheKeyFn:
    cache_key_fn = options and options.cache_key_fn
    if cache_key_fn is None:
        return default_cache_key_fn

    if not inspect.isfunction(cache_key_fn):
        raise TypeError(f"cache_key_fn must be a function: {cache_key_fn}")

    return cache_key_fn


def has_function(value: Any, fn_name: str) -> bool:
    fn = getattr(value, fn_name, None)
    return not fn or inspect.isfunction(fn)


def get_valid_cache_map(
    options: Optional[Options[Any, VT, CT]] = None
) -> Optional[CacheMap[CT, Awaitable[VT]]]:
    should_cache = not options or options.cache
    if not should_cache:
        return None

    cache_map = options and options.cache_map
    if cache_map is None:
        return Map()

    if cache_map:

        def has_no_function(fn_name: str) -> bool:
            fn = getattr(cache_map, fn_name, None)
            return not fn or inspect.isfunction(fn)

        cache_functions = ["get", "set", "delete", "clear"]
        missing_functions = list(filter(has_no_function, cache_functions))
        if len(missing_functions):
            raise TypeError(
                "Custom cache_map missing methods: " + ", ".join(missing_functions)
            )

    return cache_map
