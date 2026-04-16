"""AST-based patcher for v0.5.9 sglang container.

Injects every policy in registry.POLICY_REGISTRY into the container's:
  - server_args.py: RADIX_EVICTION_POLICY_CHOICES list
  - radix_cache.py: RadixCache.__init__ dispatch chain + strategy import

Idempotent: checks existing AST, only inserts what's missing.
Validated: re-parses both files after editing; aborts image build on failure.

Usage (inside Modal image build):
    python3 patcher.py /sgl-workspace/sglang/python/sglang/srt
"""

from __future__ import annotations

import ast
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from registry import POLICY_REGISTRY  # noqa: E402


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _validate(path: pathlib.Path) -> None:
    try:
        ast.parse(path.read_text())
    except SyntaxError as e:
        raise RuntimeError(f"Patch produced invalid Python in {path}: {e}")


def patch_server_args(path: pathlib.Path, policies: list[str]) -> list[str]:
    """Append missing policies to RADIX_EVICTION_POLICY_CHOICES. Returns added names."""
    tree = _parse(path)
    existing: set[str] = set()
    target_lineno: int | None = None
    target_end_lineno: int | None = None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "RADIX_EVICTION_POLICY_CHOICES"
            and isinstance(node.value, ast.List)
        ):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    existing.add(elt.value)
            target_lineno = node.lineno
            target_end_lineno = node.end_lineno
            break

    if target_lineno is None:
        raise RuntimeError(
            f"Could not find RADIX_EVICTION_POLICY_CHOICES assignment in {path}"
        )

    missing = [p for p in policies if p not in existing]
    if not missing:
        return []

    lines = path.read_text().splitlines(keepends=True)
    original = "".join(lines[target_lineno - 1 : target_end_lineno])
    # Rebuild as a single line: RADIX_EVICTION_POLICY_CHOICES = [...]
    new_list = sorted(existing | set(missing))
    replacement = (
        "RADIX_EVICTION_POLICY_CHOICES = ["
        + ", ".join(f'"{p}"' for p in new_list)
        + "]\n"
    )
    new_text = "".join(lines[: target_lineno - 1]) + replacement + "".join(
        lines[target_end_lineno:]
    )
    path.write_text(new_text)
    _validate(path)
    return missing


def patch_radix_cache(path: pathlib.Path, policies_with_classes: dict[str, str]) -> list[str]:
    """Add missing elif branches to RadixCache.__init__ dispatch. Returns added policy names."""
    text = path.read_text()
    tree = ast.parse(text)

    # Locate class RadixCache -> __init__ -> the if/elif chain on self.eviction_policy.
    init_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RadixCache":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_fn = item
                    break
            break
    if init_fn is None:
        raise RuntimeError(f"Could not find RadixCache.__init__ in {path}")

    # Find the top-level `if self.eviction_policy == "lru":` within __init__.
    dispatch_if: ast.If | None = None
    for stmt in init_fn.body:
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Attribute)
            and stmt.test.left.attr == "eviction_policy"
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Constant)
        ):
            dispatch_if = stmt
            break
    if dispatch_if is None:
        raise RuntimeError(
            f"Could not find eviction_policy dispatch if/elif chain in {path}"
        )

    # Walk the elif chain to find existing policy names and the terminal else (ValueError).
    existing: set[str] = set()
    cursor: ast.If = dispatch_if
    while True:
        if (
            isinstance(cursor.test, ast.Compare)
            and isinstance(cursor.test.comparators[0], ast.Constant)
        ):
            existing.add(cursor.test.comparators[0].value)
        if len(cursor.orelse) == 1 and isinstance(cursor.orelse[0], ast.If):
            cursor = cursor.orelse[0]
        else:
            break

    missing = [p for p in policies_with_classes if p not in existing]
    if not missing:
        return []

    # Splice new elif branches after the last existing elif's body, before the terminal else.
    indent = " " * dispatch_if.col_offset
    body_indent = " " * (dispatch_if.col_offset + 4)
    lines = text.splitlines(keepends=True)
    insert_after_lineno = cursor.body[-1].end_lineno  # 1-indexed, inclusive

    snippet = "".join(
        f'{indent}elif self.eviction_policy == "{name}":\n'
        f"{body_indent}self.eviction_strategy: EvictionStrategy = {policies_with_classes[name]}()\n"
        for name in missing
    )
    new_text = (
        "".join(lines[:insert_after_lineno]) + snippet + "".join(lines[insert_after_lineno:])
    )

    # Add any missing strategy-class imports to the existing evict_policy import block.
    marker = "from sglang.srt.mem_cache.evict_policy import ("
    for name in missing:
        cls = policies_with_classes[name]
        if cls in new_text.split(marker, 1)[0]:
            continue  # already imported above the dispatch
        if marker not in new_text:
            raise RuntimeError(f"Could not find evict_policy import block in {path}")
        new_text = new_text.replace(marker, f"{marker}\n    {cls},", 1)

    path.write_text(new_text)
    _validate(path)
    return missing


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: patcher.py <path-to-sglang-srt-dir>", file=sys.stderr)
        sys.exit(2)

    srt = pathlib.Path(sys.argv[1])
    server_args = srt / "server_args.py"
    radix_cache = srt / "mem_cache" / "radix_cache.py"

    for p in (server_args, radix_cache):
        if not p.exists():
            raise FileNotFoundError(p)

    policies = list(POLICY_REGISTRY.keys())
    classes = {name: info["class_name"] for name, info in POLICY_REGISTRY.items()}

    added_sa = patch_server_args(server_args, policies)
    added_rc = patch_radix_cache(radix_cache, classes)

    print(f"[patcher] server_args.py: added {added_sa or 'nothing (already present)'}")
    print(f"[patcher] radix_cache.py: added {added_rc or 'nothing (already present)'}")
    print(f"[patcher] registry: {sorted(POLICY_REGISTRY.keys())}")


if __name__ == "__main__":
    main()
