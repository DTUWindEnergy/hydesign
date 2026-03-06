from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hydesign.modular.project_builder.contracts import BuildManifest, BuildRequest
from hydesign.modular.project_builder.registry import DEFAULT_ARTIFACTS
from hydesign.modular.project_builder.source_hashes import (
    source_hash_targets_for_request,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ModuleNotFoundError:
            raise
        return yaml.safe_load(text) or {}


def write_manifest(path: Path, manifest: BuildManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def hash_sources(relative_paths: tuple[str, ...]) -> dict[str, str]:
    root = repo_root()
    hashes: dict[str, str] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def merge_artifacts(
    manifest_data: dict[str, Any] | None,
    requested_artifacts: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    existing = tuple((manifest_data or {}).get("artifacts", ()))
    merged: list[str] = []
    seen: set[str] = set()
    for name in list(existing) + list(requested_artifacts):
        if name in seen:
            continue
        merged.append(name)
        seen.add(name)
    return tuple(merged)


def merge_source_hashes(
    manifest_data: dict[str, Any] | None,
    new_hashes: dict[str, str],
) -> dict[str, str]:
    merged = dict((manifest_data or {}).get("source_hashes", {}))
    merged.update(new_hashes)
    return merged


def request_with_merged_artifacts(
    *,
    project_name: str,
    technologies: tuple[str, ...] | list[str],
    requested_artifacts: tuple[str, ...] | list[str],
    output_root: Path | str,
    enable_build: bool,
    regenerate: bool,
    builder_version: str,
) -> tuple[BuildRequest, dict[str, Any] | None]:
    initial_request = BuildRequest(
        project_name=project_name,
        technologies=technologies,
        artifacts=requested_artifacts,
        enable_build=enable_build,
        regenerate=regenerate,
        output_root=output_root,
        builder_version=builder_version,
    )
    manifest_data = load_manifest(initial_request.manifest_path)
    merged_artifacts = merge_artifacts(manifest_data, tuple(requested_artifacts))
    if merged_artifacts == initial_request.artifacts:
        return initial_request, manifest_data
    return (
        BuildRequest(
            project_name=project_name,
            technologies=technologies,
            artifacts=merged_artifacts,
            enable_build=enable_build,
            regenerate=regenerate,
            output_root=output_root,
            builder_version=builder_version,
        ),
        manifest_data,
    )


def write_package_init(output_dir: Path) -> None:
    module_names = [
        path.stem for path in output_dir.glob("*.py") if path.stem != "__init__"
    ]
    ordered: list[str] = []
    for name in DEFAULT_ARTIFACTS:
        if name in module_names:
            ordered.append(name)
    for name in sorted(module_names):
        if name not in ordered:
            ordered.append(name)
    lines = [f"from . import {name}" for name in ordered]
    lines.append("")
    lines.append(f"__all__ = {ordered!r}")
    output_dir.joinpath("__init__.py").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def collect_generated_files(output_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in output_dir.glob("*.py")))


def build_generated_artifact(
    *,
    request: BuildRequest,
    output_filename: str,
    rendered_text: str,
    built_at_utc: str,
) -> BuildManifest:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    request.output_dir.joinpath(output_filename).write_text(
        rendered_text,
        encoding="utf-8",
    )
    write_package_init(request.output_dir)
    return BuildManifest.from_request(
        request,
        generated_files=collect_generated_files(request.output_dir),
        source_hashes=hash_sources(
            source_hash_targets_for_request(request.artifacts, request.technologies)
        ),
        built_at_utc=built_at_utc,
    )
