#!/usr/bin/env python3
"""
ems_core generator - multi-file merge (refactored)
Preserves original behavior and output.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse(source: str) -> ast.Module:
    return ast.parse(source)


def _ensure_trailing_newline(s: str) -> str:
    return s if not s or s.endswith("\n") else s + "\n"


def get_source_segment(module_src: str, node: ast.AST) -> str:
    """
    Return source text for a node using ast.get_source_segment if available,
    otherwise slice by lineno/end_lineno. Ensure trailing newline.
    """
    seg = None
    try:
        seg = ast.get_source_segment(module_src, node)
    except Exception:
        seg = None

    if seg is None:
        lines = module_src.splitlines(keepends=True)
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            start = node.lineno - 1
            end = node.end_lineno
            seg = "".join(lines[start:end])
        else:
            seg = ""

    return _ensure_trailing_newline(seg)


def find_class(module: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def split_method_header_body(
    module_src: str, func_node: ast.FunctionDef
) -> Tuple[str, str]:
    """
    Extract header (decorators + def signature lines) and body (the rest) for a function node.
    Returns (header_text, body_text) with preserved newlines.
    """
    lines = module_src.splitlines(keepends=True)

    header_start = (
        func_node.decorator_list[0].lineno - 1
        if func_node.decorator_list
        else func_node.lineno - 1
    )

    if func_node.body:
        first_body_lineno = func_node.body[0].lineno - 1
    else:
        # fallback: assume body starts on the same line as def (empty function)
        first_body_lineno = func_node.lineno

    header_lines = lines[header_start:first_body_lineno]
    body_lines = lines[first_body_lineno : func_node.end_lineno]

    header_text = "".join(header_lines)
    body_text = "".join(body_lines)

    return _ensure_trailing_newline(header_text), _ensure_trailing_newline(body_text)


def normalize_body(body_text: str, indent: int = 8) -> str:
    """
    Remove common indentation from body_text and re-indent by 'indent' spaces.
    Keeps blank lines. If body is empty/whitespace, return indented 'pass\n'.
    """
    if not body_text.strip():
        return textwrap.indent("pass\n", " " * indent)

    lines = body_text.splitlines(keepends=True)

    # minimal indent among non-blank lines
    min_indent = None
    for ln in lines:
        if ln.strip() == "":
            continue
        leading = len(ln) - len(ln.lstrip())
        if min_indent is None or leading < min_indent:
            min_indent = leading
    min_indent = 0 if min_indent is None else min_indent

    normalized = []
    for ln in lines:
        if ln.strip() == "":
            normalized.append("\n")
        else:
            normalized.append(ln[min_indent:])
    return textwrap.indent("".join(normalized), " " * indent)


def collect_top_level_nodes(
    src: str, exclude_class_names: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Return (imports_list, other_top_level_list).
    Each item is the original source text for the AST node, ending with newline.
    Exclude classes whose names are in exclude_class_names.
    """
    mod = parse(src)
    imports: List[str] = []
    others: List[str] = []
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = get_source_segment(src, node)
            if seg and seg.strip():
                imports.append(_ensure_trailing_newline(seg))
        elif isinstance(node, ast.ClassDef) and node.name in exclude_class_names:
            continue
        else:
            seg = get_source_segment(src, node)
            if seg and not seg.isspace():
                others.append(_ensure_trailing_newline(seg))
    return imports, others


def extract_methods_from_class(
    module_src: str, class_node: ast.ClassDef
) -> Dict[str, Tuple[ast.FunctionDef, str, str]]:
    """
    For each FunctionDef inside class_node return a mapping:
        method_name -> (func_node, header_text, body_text)
    """
    methods: Dict[str, Tuple[ast.FunctionDef, str, str]] = {}
    for member in class_node.body:
        if isinstance(member, ast.FunctionDef):
            header, body = split_method_header_body(module_src, member)
            methods[member.name] = (member, header, body)
    return methods


def class_bases(class_node: ast.ClassDef) -> List[str]:
    res: List[str] = []
    for b in class_node.bases:
        try:
            res.append(ast.unparse(b))
        except Exception:
            if isinstance(b, ast.Name):
                res.append(b.id)
            else:
                res.append("<unknown>")
    return res


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = it.strip()
        if key not in seen:
            seen.add(key)
            out.append(_ensure_trailing_newline(it))
    return out


def build_combined_class_text_multi(
    combined_class_name: str,
    srcs: List[str],
    cls_nodes: List[ast.ClassDef],
    src_files: List[str],
    prefer_bases_from_first: bool = True,
) -> str:
    """
    Build combined class text from multiple classes.
    See original docstring for semantics.
    """
    if not cls_nodes:
        raise RuntimeError("No class nodes provided")

    all_methods_per_file: List[Dict[str, Tuple[ast.FunctionDef, str, str]]] = [
        extract_methods_from_class(src, cls) for src, cls in zip(srcs, cls_nodes)
    ]

    # ordered method names: primary file order, then others (file order & within-file order)
    ordered: List[str] = []
    primary_cls = cls_nodes[0]
    primary_methods = all_methods_per_file[0]
    for m in primary_cls.body:
        if isinstance(m, ast.FunctionDef) and m.name in primary_methods:
            ordered.append(m.name)

    for file_idx in range(1, len(cls_nodes)):
        cls = cls_nodes[file_idx]
        methods_map = all_methods_per_file[file_idx]
        for m in cls.body:
            if (
                isinstance(m, ast.FunctionDef)
                and m.name in methods_map
                and m.name not in ordered
            ):
                ordered.append(m.name)

    # determine bases
    bases: Optional[List[str]] = None
    b_first = class_bases(cls_nodes[0])
    if prefer_bases_from_first and b_first:
        bases = b_first
    else:
        for cls in cls_nodes[1:]:
            b = class_bases(cls)
            if b:
                bases = b
                break
    if not bases:
        bases = ["EMSCoreInterface"]

    bases_text = "(" + ", ".join(bases) + ")"

    parts: List[str] = []
    parts.append(f"class {combined_class_name}{bases_text}:\n")
    parts.append(
        '    """Auto-generated combined class (merged bodies of multiple classes)\n\n'
    )
    parts.append("    NOTE: Please review for semantic issues.\n")
    parts.append('    """\n\n')

    if not ordered:
        parts.append("    pass\n")
        return "".join(parts)

    methods_maps = {
        "solve",
        "batch",
        "cplex",
        "pyomo",
        "ortools",
        "update_final_outputs",
    }
    #
    print(ordered)
    ordered = [
        "__init__",
        "solve",
        "batch_data",
        "cplex_variables",
        "cplex_constraints",
        "cplex_opjective",
        "cplex_solve",
        "cplex_outputs",
        "pyomo_variables",
        "pyomo_constraints",
        "pyomo_opjective",
        "pyomo_solve",
        "pyomo_outputs",
        "ortools_variables",
        "ortools_constraints",
        "ortools_objective",
        "ortools_solve",
        "ortools_outputs",
        "update_final_outputs",
    ]
    for mname in ordered:
        # print(mname)

        # choose header from first file that has it (prefer primary)
        chosen_header = ""
        if mname in {
            "__init__",
            "solve",
            "final_process",
        }:
            chosen_header = f"    def {mname}(self, **kwargs):"

        elif mname == "batch_data":
            chosen_header = f"    def {mname}(self, batch):"

        elif mname == "update_final_outputs":
            chosen_header = f"    def {mname}(self, out):"
        else:
            prefix = mname.split("_", 1)[0]
            if prefix in methods_maps:
                # print("True")
                # print(prefix)
                # print(methods_maps)

                chosen_header = f"    def {prefix}(self, **kwarg):"
                # print(methods_maps)
                methods_maps.remove(prefix)
            # break
        chosen_header = _ensure_trailing_newline(chosen_header).rstrip("\n") + "\n"
        # print(chosen_header)
        parts.append(chosen_header)

        appended_any = False
        for idx, methods_map in enumerate(all_methods_per_file):
            if mname not in methods_map:
                continue
            _, _, body_text = methods_map[mname]
            norm = normalize_body(body_text, indent=8)
            if appended_any:
                sep = f"\n# --- content from {os.path.basename(src_files[idx])} ---\n"
                parts.append(textwrap.indent(sep, " " * 8))
            parts.append(norm)
            appended_any = True

        if not appended_any:
            parts.append(textwrap.indent("pass\n", " " * 8))

        parts.append("\n")  # separation between methods

    return "".join(parts)


def merge_multiple_files(
    files: List[str],
    classes: List[str],
    out_file: str,
    combined_name: str = "EMSCore",
) -> None:
    """
    Merge multiple files/classes into a single module written to out_file.
    Preserves original behavior.
    """
    if len(files) != len(classes):
        raise RuntimeError("files and classes lists must have same length")

    srcs = [read_text(p) for p in files]
    mods = [parse(s) for s in srcs]

    cls_nodes: List[ast.ClassDef] = []
    for mod, cls_name, f in zip(mods, classes, files):
        cls = find_class(mod, cls_name)
        if cls is None:
            raise RuntimeError(f"Class {cls_name} not found in {f}")
        cls_nodes.append(cls)

    # collect imports and other top-level nodes across all files, excluding the classes being merged
    all_imports: List[str] = []
    all_others: List[str] = []
    for src, cls_name in zip(srcs, classes):
        imp, oth = collect_top_level_nodes(src, exclude_class_names=[cls_name])
        all_imports.extend(imp)
        all_others.extend(oth)

    imports_merged = dedupe_keep_order(all_imports)
    others_merged = all_others

    combined_class_src = build_combined_class_text_multi(
        combined_name, srcs, cls_nodes, files
    )

    out_parts: List[str] = []
    out_parts.append("# Auto-generated combined module\n")
    out_parts.append(
        f"# Generated from: {', '.join(os.path.basename(p) for p in files)}\n\n"
    )

    if imports_merged:
        out_parts.extend(imports_merged)
        out_parts.append("\n")

    for item in others_merged:
        out_parts.append(item)
    if others_merged:
        out_parts.append("\n")

    out_parts.append(combined_class_src)

    module_text = "".join(out_parts)
    module_text = _ensure_trailing_newline(module_text)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(module_text)

    print(f"Combined module written to: {out_file}")


def _to_pascal_case(s: str) -> str:
    """Convert strings like 'wind', 'pv_system', 'solar-park' -> 'Wind', 'PvSystem', 'SolarPark'."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", s) if p]
    return "".join(p.capitalize() for p in parts)


def generate_ems_core(
    technologies: Iterable[str], raise_on_missing: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Build file & class lists from the given technologies and call merge_multiple_files().
    Returns (files, classes) used to create the combined file.
    """
    MODULES_DIR = Path(__file__).parent
    DEFAULT_FILE = MODULES_DIR / "ems_core_defaults.py"
    OUT_FILE = MODULES_DIR.parent / "ems_core.py"

    files: List[str] = [str(DEFAULT_FILE)]
    classes: List[str] = ["EMSCoreDefaults"]

    missing: List[Tuple[str, Path]] = []
    for tech in technologies:
        tech = str(tech).strip()
        if not tech:
            continue
        fname = f"ems_core_{tech}.py"
        fpath = MODULES_DIR / fname
        classname = "EMSCore" + _to_pascal_case(tech)
        if not fpath.exists():
            missing.append((tech, fpath))
            if raise_on_missing:
                raise FileNotFoundError(
                    f"Technology file for '{tech}' not found at {fpath}"
                )
            else:
                print(
                    f"[generate_ems_core_combined] skipping missing tech '{tech}' (expected {fpath})"
                )
                continue
        files.append(str(fpath))
        classes.append(classname)

    if missing and not raise_on_missing and len(files) == 1:
        print(
            "[generate_ems_core_combined] Warning: only defaults included (no technology files found)."
        )

    merge_multiple_files(files, classes, str(OUT_FILE), combined_name="EMSCore")
    return files, classes


if __name__ == "__main__":
    generate_ems_core_combined(technologies=("wind", "solar", "battery"))
