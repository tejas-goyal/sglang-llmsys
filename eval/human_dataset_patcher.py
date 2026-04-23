"""AST-based patcher for v0.5.9 bench_serving.py.

Rewrites `gen_prompt` to dispatch to `_gen_prompt_human` when
`SGLANG_EVAL_GEN_PROMPT_SOURCE=human`, and suffixes the
generated-shared-prefix pickle cache key with the same env var so synthetic
vs human runs can't collide on the on-disk pickle cache.

Idempotent: checks for a sentinel marker, only patches if absent.
Validated: re-parses the file after editing; aborts image build on failure.

Usage (inside Modal image build):
    python3 human_dataset_patcher.py /sgl-workspace/sglang/python/sglang/bench_serving.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

MARKER = "# SGLANG_EVAL_HUMAN_PATCHED"


_HUMAN_HELPER = '''# SGLANG_EVAL_HUMAN_PATCHED
_HUMAN_CORPUS_TOKENS: dict = {}


def _gen_prompt_human(tokenizer, token_num):
    """Sample a contiguous real-text token window from a corpus file.

    Corpus path is read from ``SGLANG_EVAL_HUMAN_CORPUS``. The corpus is
    tokenized once (memoized on ``id(tokenizer)``) so subsequent calls are
    cheap. ``random.randint`` picks the start offset, so determinism is
    inherited from whatever ``random.seed(...)`` the caller already set.
    """
    path = os.environ.get(
        "SGLANG_EVAL_HUMAN_CORPUS",
        "/root/.cache/huggingface/eval/human_corpus.txt",
    )
    key = id(tokenizer)
    cached = _HUMAN_CORPUS_TOKENS.get(key)
    if cached is None:
        with open(path, "r") as _f:
            text = _f.read()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < max(token_num, 1024):
            raise RuntimeError(
                f"Human corpus at {path} has only {len(tokens)} tokens; "
                f"not enough for gen_prompt(token_num={token_num})"
            )
        _HUMAN_CORPUS_TOKENS[key] = tokens
        cached = tokens
    if token_num <= 0:
        return ""
    max_start = len(cached) - token_num
    start = random.randint(0, max_start)
    return tokenizer.decode(cached[start : start + token_num])


'''


_GEN_PROMPT_REPLACEMENT = '''def gen_prompt(tokenizer, token_num):
    """Generate a prompt of specified token length (random or human-readable)."""
    if os.environ.get("SGLANG_EVAL_GEN_PROMPT_SOURCE") == "human":
        return _gen_prompt_human(tokenizer, token_num)
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)
'''


_CACHE_PATH_REPLACEMENT = '''def get_gen_prefix_cache_path(args, tokenizer):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"
    _suffix = os.environ.get("SGLANG_EVAL_GEN_PROMPT_SOURCE", "random")
    cache_key = (
        f"gen_shared_prefix_{args.seed}_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
        f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_{args.gsp_output_len}_"
        f"{tokenizer.__class__.__name__}_{_suffix}.pkl"
    )
    return cache_dir / cache_key
'''


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"Could not find top-level function {name!r}")


def _validate(text: str, path: pathlib.Path) -> None:
    try:
        ast.parse(text)
    except SyntaxError as e:
        raise RuntimeError(f"Patch produced invalid Python in {path}: {e}")


def patch(path: pathlib.Path) -> list[str]:
    text = path.read_text()
    if MARKER in text:
        return []

    # Pass 1: splice helper + new gen_prompt over the existing gen_prompt span.
    tree = ast.parse(text)
    gp = _find_func(tree, "gen_prompt")
    lines = text.splitlines(keepends=True)
    new_lines = (
        lines[: gp.lineno - 1]
        + [_HUMAN_HELPER + _GEN_PROMPT_REPLACEMENT]
        + lines[gp.end_lineno :]
    )
    mid_text = "".join(new_lines)

    # Pass 2: rewrite get_gen_prefix_cache_path on the shifted file.
    tree2 = ast.parse(mid_text)
    cache_fn = _find_func(tree2, "get_gen_prefix_cache_path")
    lines2 = mid_text.splitlines(keepends=True)
    new_lines2 = (
        lines2[: cache_fn.lineno - 1]
        + [_CACHE_PATH_REPLACEMENT]
        + lines2[cache_fn.end_lineno :]
    )
    final_text = "".join(new_lines2)

    _validate(final_text, path)
    path.write_text(final_text)
    return ["gen_prompt", "get_gen_prefix_cache_path"]


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: human_dataset_patcher.py <path-to-bench_serving.py>",
            file=sys.stderr,
        )
        sys.exit(2)

    target = pathlib.Path(sys.argv[1])
    if not target.exists():
        raise FileNotFoundError(target)

    patched = patch(target)
    if patched:
        print(f"[human_dataset_patcher] patched: {patched}")
    else:
        print("[human_dataset_patcher] already patched; no-op")


if __name__ == "__main__":
    main()
