"""Registry by name of the pluggable components (CONVENTIONS.md §4.1).

A component registers itself through a decorator; the pipeline only knows names.
No `if approach == ...` may exist anywhere else in the project.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from types import ModuleType
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> object table, refusing silent duplicates."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Registration decorator. The name MUST be that of the matching YAML file."""

        def decorator(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise KeyError(
                    f"'{name}' is already registered in the '{self.namespace}' registry. "
                    "Choose another name rather than overwriting."
                )
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Return a registered component, or fail with the list of valid names."""
        if name not in self._items:
            raise KeyError(
                f"'{name}' not found in the '{self.namespace}' registry. "
                f"Available: {sorted(self._items)}. "
                "Check that the module is imported (see load_plugins)."
            )
        return self._items[name]

    def available(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


APPROACHES: Registry[Any] = Registry("approach")
METRICS: Registry[Any] = Registry("metric")
ADAPTERS: Registry[Any] = Registry("adapter")

register_approach = APPROACHES.register
register_metric = METRICS.register
register_adapter = ADAPTERS.register


def load_plugins(package: str) -> list[str]:
    """Import every sub-module of a package to trigger the registrations.

    Side effect: Python imports only, no disk write.
    """
    module: ModuleType = importlib.import_module(package)
    loaded: list[str] = []
    for info in pkgutil.iter_modules(module.__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{info.name}")
        loaded.append(info.name)
    return loaded


# Entry-point group of the external plugins (the `research/` package, a user's own
# approaches). Each entry point names a module that may define:
#   register()   imports its components, which register themselves here
#   CONFIG_DIR   folder of extra Hydra configs (approach/, experiment/ ...)
PLUGIN_GROUP = "extraits.pose_plugins"


def plugin_modules() -> list[ModuleType]:
    """Modules of the installed plugins (entry points of PLUGIN_GROUP)."""
    from importlib.metadata import entry_points

    return [ep.load() for ep in entry_points(group=PLUGIN_GROUP)]


def load_all_plugins() -> None:
    """Load approaches, metrics and adapters, then the installed plugins.

    Called once when the CLI starts.
    """
    load_plugins("extraits.pose.approaches")
    load_plugins("extraits.pose.evaluation.metrics")
    load_plugins("extraits.pose.data.adapters")
    for module in plugin_modules():
        register = getattr(module, "register", None)
        if callable(register):
            register()
