"""Utilities for hierarchical profiling across the ViPE codebase."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Callable, Dict, Iterable, Iterator, TypeVar, cast


@dataclass(slots=True)
class _StackEntry:
    node: "ProfileNode"
    start_time: float
    child_time: float = 0.0


@dataclass(slots=True)
class ProfileNode:
    name: str
    calls: int = 0
    total_time: float = 0.0
    self_time: float = 0.0
    children: Dict[str, "ProfileNode"] = field(default_factory=dict)

    def get_child(self, name: str) -> "ProfileNode":
        if name not in self.children:
            self.children[name] = ProfileNode(name)
        return self.children[name]


class HierarchicalProfiler:
    """Simple hierarchical profiler based on wall clock time."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.root = ProfileNode("root")
        self._stack: list[_StackEntry] = []

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        self.root = ProfileNode("root")
        self._stack.clear()

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        parent = self._stack[-1].node if self._stack else self.root
        node = parent.get_child(name)
        entry = _StackEntry(node=node, start_time=perf_counter())
        self._stack.append(entry)
        try:
            yield
        finally:
            end_time = perf_counter()
            duration = end_time - entry.start_time
            node.calls += 1
            node.total_time += duration
            exclusive = duration - entry.child_time
            if exclusive < 0.0:
                exclusive = 0.0
            node.self_time += exclusive
            self._stack.pop()
            if self._stack:
                self._stack[-1].child_time += duration
            else:
                # Update root timings when unwinding the outer-most section
                self.root.total_time += duration

    def iter_nodes(self) -> Iterable[ProfileNode]:
        return self.root.children.values()

    def _format_node(
        self,
        node: ProfileNode,
        *,
        depth: int,
        parent_total: float,
        total_root: float,
        lines: list[str],
        min_percentage: float,
        max_depth: int | None,
        prefix_parts: list[str] | None = None,
        is_last: bool = True,
    ) -> None:
        if max_depth is not None and depth > max_depth:
            return

        if parent_total <= 0.0:
            parent_total = node.total_time or total_root or 1.0

        percent_of_parent = (node.total_time / parent_total * 100.0) if parent_total else 0.0
        percent_of_total = (node.total_time / total_root * 100.0) if total_root else 0.0

        if percent_of_total < min_percentage:
            return

        if prefix_parts is None:
            prefix_parts = []

        branch = "└── " if is_last else "├── "
        indent = "".join(prefix_parts) + branch

        # Extra spacing for readability
        name = f"{indent}{node.name}"
        name = (name[:67] + "…") if len(name) > 70 else name

        avg_time = node.total_time / node.calls if node.calls > 0 else 0.0

        lines.append(
            "{:<70} {:>8} {:>12.3f} {:>12.3f} {:>12.3f} {:>9.2f}% {:>9.2f}%".format(
                name, node.calls, node.total_time, node.self_time, avg_time,
                percent_of_parent, percent_of_total
            )
        )

        children = sorted(node.children.values(), key=lambda child: child.total_time, reverse=True)
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            new_prefix_parts = prefix_parts.copy()
            if depth > 0:
                if is_last:
                    new_prefix_parts.append("    ")
                else:
                    new_prefix_parts.append("│   ")
            self._format_node(
                child,
                depth=depth + 1,
                parent_total=node.total_time,
                total_root=total_root,
                lines=lines,
                min_percentage=min_percentage,
                max_depth=max_depth,
                prefix_parts=new_prefix_parts,
                is_last=child_is_last,
            )

    def report(self, *, min_percentage: float = 0.0, max_depth: int | None = None) -> str:
        if not self.enabled:
            return "Profiler disabled."

        total_root = max(
            self.root.total_time, 
            sum(child.total_time for child in self.root.children.values())
        )

        header = "{:<70} {:>8} {:>12} {:>12} {:>12} {:>9} {:>9}".format(
            "Section", "Calls", "Total (s)", "Self (s)", "Avg (s)", "% Parent", "% Total"
        )
        separator = "=" * len(header)
        lines = [header, separator]

        children = sorted(self.root.children.values(), key=lambda child: child.total_time, reverse=True)
        for child in children:
            self._format_node(
                child,
                depth=0,
                parent_total=total_root,
                total_root=total_root,
                lines=lines,
                min_percentage=min_percentage,
                max_depth=max_depth,
            )

        return "\n".join(lines)




_GLOBAL_PROFILER = HierarchicalProfiler()


def get_profiler() -> HierarchicalProfiler:
    return _GLOBAL_PROFILER


@contextmanager
def profiler_section(name: str) -> Iterator[None]:
    with get_profiler().profile(name):
        yield


F = TypeVar("F", bound=Callable[..., object])


def profile_function(name: str | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        label = name or func.__qualname__

        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            if not profiler.enabled:
                return func(*args, **kwargs)
            with profiler.profile(label):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
