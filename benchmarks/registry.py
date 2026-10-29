"""Dataset registry — maps names to DatasetABC instances."""

from benchmarks.base import DatasetABC

_REGISTRY: dict[str, DatasetABC] = {}


def register(name: str):
    """Decorator to register a DatasetABC subclass under ``name``."""
    def wrapper(cls):
        _REGISTRY[name] = cls()
        return cls
    return wrapper


def get_dataset(name: str) -> DatasetABC:
    if name not in _REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered. Available: {list_datasets()}")
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    return sorted(_REGISTRY.keys())
