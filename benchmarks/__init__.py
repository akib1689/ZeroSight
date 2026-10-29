import benchmarks.bird.dataset  # noqa: F401
import benchmarks.employees.dataset  # noqa: F401
import benchmarks.fever.dataset  # noqa: F401
import benchmarks.hitab.dataset  # noqa: F401
import benchmarks.hotpotqa.dataset  # noqa: F401 — triggers @register
import benchmarks.lindisfarne.dataset  # noqa: F401
import benchmarks.musique.dataset  # noqa: F401
import benchmarks.spider.dataset  # noqa: F401
import benchmarks.wiki2.dataset  # noqa: F401
from benchmarks.base import DatasetABC, Task
from benchmarks.registry import get_dataset, list_datasets

__all__ = ["DatasetABC", "Task", "get_dataset", "list_datasets"]
