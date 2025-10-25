#!/usr/bin/env python3
"""
ems_core generator - multi-file merge
"""
from __future__ import annotations

import ast
import os
import textwrap
from typing import Dict, List, Optional, Tuple


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse(source: str) -> ast.Module:
    return ast.parse(source)


def get_source_segment(module_src: str, node: ast.AST) -> str:
    """
    Return source text for a node using ast.get_source_segment if available,
    otherwise slice by lineno/end_lineno. Ensure trailing newline.
    """
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
    if seg and not seg.endswith("\n"):
        seg += "\n"
    return seg


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
    Uses module source lines so lineno/end_lineno are module-relative.
    Returns (header_text, body_text) with preserved newlines.
    """
    lines = module_src.splitlines(keepends=True)
    # header starts at first decorator if any, else at def.lineno
    header_start = (
        func_node.decorator_list[0].lineno - 1
        if func_node.decorator_list
        else func_node.lineno - 1
    )

    # find first body line lineno (first statement inside function)
    if func_node.body:
        first_body_lineno = func_node.body[0].lineno - 1
    else:
        first_body_lineno = func_node.lineno  # fallback

    header_lines = lines[header_start:first_body_lineno]
    body_lines = lines[first_body_lineno : func_node.end_lineno]

    header_text = "".join(header_lines)
    body_text = "".join(body_lines)

    # Ensure trailing newline
    if header_text and not header_text.endswith("\n"):
        header_text += "\n"
    if body_text and not body_text.endswith("\n"):
        body_text += "\n"
    return header_text, body_text


def normalize_body(body_text: str, indent: int = 8) -> str:
    """
    Remove common indentation from body_text and re-indent by 'indent' spaces.
    Keeps blank lines.
    """
    if not body_text.strip():
        return textwrap.indent("pass\n", " " * indent)

    lines = body_text.splitlines(keepends=True)
    # compute minimal indent among non-blank lines
    min_indent = None
    for ln in lines:
        if ln.strip() == "":
            continue
        leading = len(ln) - len(ln.lstrip())
        if min_indent is None or leading < min_indent:
            min_indent = leading
    if min_indent is None:
        min_indent = 0

    stripped = []
    for ln in lines:
        if ln.strip() == "":
            stripped.append("\n")
        else:
            stripped.append(ln[min_indent:])
    joined = "".join(stripped)
    return textwrap.indent(joined, " " * indent)


def collect_top_level_nodes(
    src: str, exclude_class_names: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Return (imports_list, other_top_level_list).
    Each item is the original source text for the AST node, ending with newline.
    Exclude classes whose names are in exclude_class_names.
    """
    mod = parse(src)
    imports = []
    others = []
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = get_source_segment(src, node)
            if seg and seg.strip():
                imports.append(seg if seg.endswith("\n") else seg + "\n")
        elif isinstance(node, ast.ClassDef) and node.name in exclude_class_names:
            continue
        else:
            seg = get_source_segment(src, node)
            if seg and not seg.isspace():
                others.append(seg if seg.endswith("\n") else seg + "\n")
    return imports, others


def extract_methods_from_class(
    module_src: str, class_node: ast.ClassDef
) -> Dict[str, Tuple[ast.FunctionDef, str, str]]:
    """
    For each FunctionDef inside class_node return a mapping:
        method_name -> (func_node, header_text, body_text)
    header/body extracted from module_src using node lineno fields.
    We skip non-FunctionDef class body elements (assignments etc.) — they are not included in the combined class.
    """
    methods = {}
    for member in class_node.body:
        if isinstance(member, ast.FunctionDef):
            header, body = split_method_header_body(module_src, member)
            methods[member.name] = (member, header, body)
    return methods


def class_bases(class_node: ast.ClassDef) -> List[str]:
    res = []
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
    out = []
    for it in items:
        key = it.strip()
        if key not in seen:
            seen.add(key)
            out.append(it if it.endswith("\n") else it + "\n")
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
    - srcs: list of module sources corresponding to cls_nodes
    - cls_nodes: list of ast.ClassDef objects
    - src_files: list of file paths (for separator comments)
    The first class is treated as primary: its method order and headers are preferred.
    Methods present only in later classes are appended after primary-class methods (in order of first appearance).
    For each method present in multiple classes, the output contains primary body first (if present),
    then bodies from subsequent classes (in subsequent-file order), with separators.
    """
    if not cls_nodes:
        raise RuntimeError("No class nodes provided")

    # extract methods for each class
    all_methods_per_file: List[Dict[str, Tuple[ast.FunctionDef, str, str]]] = []
    for src, cls in zip(srcs, cls_nodes):
        all_methods_per_file.append(extract_methods_from_class(src, cls))

    # Build ordered method list:
    ordered: List[str] = []
    # 1) methods from primary (first) class in original order
    primary_cls = cls_nodes[0]
    primary_methods = all_methods_per_file[0]
    for m in primary_cls.body:
        if isinstance(m, ast.FunctionDef) and m.name in primary_methods:
            ordered.append(m.name)
    # 2) methods from the rest (in order of files and within-file order) that are not yet included
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
    b_first = class_bases(cls_nodes[0])
    bases = None
    if prefer_bases_from_first and b_first:
        bases = b_first
    else:
        # find first non-empty bases among all classes
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
    parts.append(
        "    NOTE: bodies are concatenated (primary class then others). Please review for semantic issues.\n"
    )
    parts.append('    """\n\n')

    if not ordered:
        parts.append("    pass\n")
        return "".join(parts)

    # for each method name, gather header (prefer primary), then append bodies from each file that has it
    for mname in ordered:
        # choose header from first file that contains the method (prefer primary)
        chosen_header = ""
        header_source_file_idx = None
        for idx, methods_map in enumerate(all_methods_per_file):
            if mname in methods_map:
                _, h, _ = methods_map[mname]
                chosen_header = h
                header_source_file_idx = idx
                break
        # ensure header formatting
        chosen_header = chosen_header.rstrip("\n") + "\n"
        chosen_header = textwrap.indent(chosen_header, " " * 0)

        parts.append(chosen_header)

        # for each file, append its normalized body if present
        any_body = False
        for idx, methods_map in enumerate(all_methods_per_file):
            if mname in methods_map:
                any_body = True
                _, _, body_text = methods_map[mname]
                norm = normalize_body(body_text, indent=8)
                # If this is not the first included body, insert a separator comment
                first_included = idx == 0 and (mname in all_methods_per_file[0])
                # We want to include bodies in order: primary first, then others.
                # So don't special-case: just append in file order; separator before bodies from file_idx >=1 when there was already an earlier body appended.
                # Determine if we should add a separator: if this is not the first body appended for this method
                # To track that, check whether we've already appended any non-empty body for this method so far (we use any_body_appended flag)
                # We'll compute via temporary variable:
                break  # break to re-append bodies with explicit logic below

        # now append bodies in order and add separators between them
        appended_any = False
        for idx, methods_map in enumerate(all_methods_per_file):
            if mname not in methods_map:
                continue
            _, _, body_text = methods_map[mname]
            norm = normalize_body(body_text, indent=8)
            if appended_any:
                sep = f"\n# --- body from {os.path.basename(src_files[idx])} ---\n"
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
    combined_name: str = "EMSCoreCombined",
) -> None:
    """
    Merge multiple files/classes into a single module written to out_file.
    - files: list of file paths (must be same length as classes)
    - classes: list of class names (same length as files)
    The first file/class is primary; files[1:] are treated as 'secondary' and appended in order.
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
    all_imports = []
    all_others = []
    for src, cls_name in zip(srcs, classes):
        imp, oth = collect_top_level_nodes(src, exclude_class_names=[cls_name])
        all_imports.extend(imp)
        all_others.extend(oth)

    imports_merged = dedupe_keep_order(all_imports)
    # keep other top-level nodes in order (you might dedupe if you want)
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
    if not module_text.endswith("\n"):
        module_text += "\n"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(module_text)

    print(f"Combined module written to: {out_file}")


def main():
    # Example usage: treat file1 as primary, files 2..N are appended (treated like second file)
    files = [
        r"C:\code\hydesign\hydesign\ems\ems_core_defaults.py",
        r"C:\code\hydesign\hydesign\ems\ems_core_wind.py",
        r"C:\code\hydesign\hydesign\ems\ems_core_solar.py",
        r"C:\code\hydesign\hydesign\ems\ems_core_battery.py",
        # add as many additional files as you want below; they will be treated like the "second" file
        # r"C:\code\hydesign\hydesign\ems\ems_core_extra.py",
    ]
    classes = [
        "EMSCoreDefaults",
        "EMSCoreWind",
        "EMSCoreSolar",
        "EMSCoreBattery",
        # corresponding class names...
        # "EMSCoreExtra",
    ]
    out = r"C:\code\hydesign\hydesign\ems\ems_core_combined.py"
    combined_name = "EMSCoreCombined"

    merge_multiple_files(files, classes, out, combined_name)


if __name__ == "__main__":
    main()
