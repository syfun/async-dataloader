import pytest

from dataloader import DataLoader

pytestmark = pytest.mark.asyncio


async def batch_fn(keys):
    return keys


@pytest.fixture()
def loader(scope='function'):
    return DataLoader(batch_fn)


async def test_load(loader):
    k1 = loader.load(1)
    k2 = loader.load(2)
    assert await k1 == 1
    assert await k2 == 2

    assert await loader.load(3) == 3


async def test_load_many(loader):
    assert await loader.load_many([1, 2]) == [1, 2]
    assert await loader.load_many([]) == []


async def test_cache(loader):
    k1 = loader.load(1)
    k2 = loader.load(2)
    k3 = loader.load(1)
    assert k1 == k3
    assert k1 != k2

    assert await k1 == 1
    assert await k2 == 2
    assert await k3 == 1

    loader.clear(1)
    k4 = loader.load(1)
    assert k1 != k4
    assert await k4 == 1

    loader.clear_all()
    k5 = loader.load(2)
    assert k2 != k5
    assert await k5 == 2


async def test_prime(loader):
    error1 = TypeError(1)
    loader.prime(1, error1)
    loader.prime(2, 2)

    k1 = loader.load(1)
    k2 = loader.load(2)

    assert k1.exception() == error1
    assert k2.result() == 2

    assert await k2 == 2
    with pytest.raises(TypeError):
        await k1

    assert await loader.load(3) == 3


async def test_error_batch_fn():
    def batch_fn1(keys):
        return keys

    loader = DataLoader(batch_fn1)
    with pytest.raises(TypeError):
        await loader.load(1)

    def batch_fn2(keys):
        return 1

    loader = DataLoader(batch_fn2)
    with pytest.raises(TypeError):
        await loader.load(1)

    def batch_fn3(keys):
        return []

    loader = DataLoader(batch_fn3)
    with pytest.raises(TypeError):
        await loader.load(1)
