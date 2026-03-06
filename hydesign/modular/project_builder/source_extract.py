from __future__ import annotations

import ast
from pathlib import Path


def _ensure_trailing_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def _get_source_segment(module_src: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(module_src, node)
    if segment is not None:
        return _ensure_trailing_newline(segment)
    lines = module_src.splitlines(keepends=True)
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        return _ensure_trailing_newline(
            "".join(lines[node.lineno - 1 : node.end_lineno])
        )
    return ""


def extract_symbols_module(
    *,
    source_path: Path,
    symbol_names: tuple[str, ...] | list[str],
    header_lines: tuple[str, ...] | list[str] = (),
) -> str:
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    imports: list[str] = []
    definitions: dict[str, str] = {}

    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_get_source_segment(source, node))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = _get_source_segment(source, node)

    missing = [name for name in symbol_names if name not in definitions]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing symbols in {source_path}: {missing_str}")

    parts: list[str] = []
    if header_lines:
        parts.extend(_ensure_trailing_newline(line) for line in header_lines)
        parts.append("\n")
    parts.extend(imports)
    parts.append("\n")
    for name in symbol_names:
        parts.append(definitions[name])
        parts.append("\n")
    return "".join(parts).rstrip() + "\n"


def extract_symbol_definitions(
    *,
    source_path: Path,
    symbol_names: tuple[str, ...] | list[str],
) -> dict[str, str]:
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    definitions: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in symbol_names:
                definitions[node.name] = _get_source_segment(source, node)

    missing = [name for name in symbol_names if name not in definitions]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing symbols in {source_path}: {missing_str}")
    return definitions


def render_full_module(
    *,
    source_path: Path,
    header_lines: tuple[str, ...] | list[str] = (),
    replacements: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
) -> str:
    source = source_path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in source:
            raise ValueError(
                "Full-module rendering rewrite failed because the expected source "
                f"fragment was not found in {source_path}: {old!r}"
            )
        source = source.replace(old, new, 1)

    parts: list[str] = []
    if header_lines:
        parts.extend(_ensure_trailing_newline(line) for line in header_lines)
        parts.append("\n")
    parts.append(_ensure_trailing_newline(source))
    return "".join(parts).rstrip() + "\n"
