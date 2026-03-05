from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _normalize_names(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip().lower().replace("-", "_")
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _slugify(value: str) -> str:
    chars: list[str] = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


@dataclass(frozen=True)
class ArtifactContract:
    """Contract for a generated project artifact such as assembly or EMS."""

    name: str
    output_filename: str
    source_package: str
    entry_symbol: str
    description: str


@dataclass(frozen=True)
class BuildRequest:
    """User-facing request for creating or reusing a combined project."""

    project_name: str
    technologies: tuple[str, ...] | list[str]
    artifacts: tuple[str, ...] | list[str]
    enable_build: bool = True
    regenerate: bool = False
    output_root: Path | str = Path("hydesign") / "combined_projects"
    builder_version: str = "0.1"

    def __post_init__(self) -> None:
        technologies = _normalize_names(self.technologies)
        if not technologies:
            raise ValueError("BuildRequest requires at least one technology.")

        artifacts = _normalize_names(self.artifacts)
        if not artifacts:
            raise ValueError("BuildRequest requires at least one artifact.")

        project_name = self.project_name.strip() or "_".join(technologies)
        project_slug = _slugify(project_name)
        if not project_slug:
            raise ValueError("BuildRequest project name resolves to an empty slug.")

        object.__setattr__(self, "project_name", project_name)
        object.__setattr__(self, "project_slug", project_slug)
        object.__setattr__(self, "technologies", technologies)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.project_slug

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.yml"

    @property
    def package_name(self) -> str:
        return self.project_slug


@dataclass(frozen=True)
class BuildManifest:
    """Minimal manifest written next to generated project files."""

    project_name: str
    project_slug: str
    technologies: tuple[str, ...]
    artifacts: tuple[str, ...]
    builder_version: str
    generated_files: tuple[str, ...] = ()
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    built_at_utc: str | None = None

    @classmethod
    def from_request(
        cls,
        request: BuildRequest,
        *,
        generated_files: tuple[str, ...] | list[str] = (),
        source_hashes: Mapping[str, str] | None = None,
        built_at_utc: str | None = None,
    ) -> "BuildManifest":
        return cls(
            project_name=request.project_name,
            project_slug=request.project_slug,
            technologies=request.technologies,
            artifacts=request.artifacts,
            builder_version=request.builder_version,
            generated_files=tuple(generated_files),
            source_hashes=dict(source_hashes or {}),
            built_at_utc=built_at_utc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "technologies": list(self.technologies),
            "artifacts": list(self.artifacts),
            "builder_version": self.builder_version,
            "generated_files": list(self.generated_files),
            "source_hashes": dict(self.source_hashes),
            "built_at_utc": self.built_at_utc,
        }
