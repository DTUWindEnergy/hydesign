from __future__ import annotations

from typing import Any, Mapping

from hydesign.modular.project_builder.contracts import BuildManifest, BuildRequest


def _manifest_value(manifest: BuildManifest | Mapping[str, Any], key: str) -> Any:
    if isinstance(manifest, BuildManifest):
        return getattr(manifest, key)
    return manifest.get(key)


def manifest_matches_request(
    manifest: BuildManifest | Mapping[str, Any],
    request: BuildRequest,
) -> bool:
    """Return True when an existing manifest matches the requested build."""

    technologies = tuple(_manifest_value(manifest, "technologies") or ())
    artifacts = tuple(_manifest_value(manifest, "artifacts") or ())
    project_slug = _manifest_value(manifest, "project_slug")
    builder_version = _manifest_value(manifest, "builder_version")

    return (
        project_slug == request.project_slug
        and technologies == request.technologies
        and artifacts == request.artifacts
        and builder_version == request.builder_version
    )


def should_build(
    request: BuildRequest,
    manifest: BuildManifest | Mapping[str, Any] | None = None,
) -> bool:
    """Apply the agreed build toggle semantics."""

    if not request.enable_build:
        return False
    if request.regenerate:
        return True
    if manifest is None:
        return True
    return not manifest_matches_request(manifest, request)
