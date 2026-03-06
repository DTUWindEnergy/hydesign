import importlib
import shutil
import sys
from pathlib import Path

from hydesign.modular import build_baseline_wsb_assembly_project


def _project_root(project_name: str) -> Path:
    return (Path.cwd() / "hydesign" / "modular" / "projects" / project_name).resolve()


def _drop_generated_modules(project_name: str) -> None:
    prefix = f"hydesign.modular.projects.{project_name}"
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(prefix + "."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def test_generated_project_imports_from_modular_namespace():
    project_name = "pytest_modular_namespace"
    output_dir = _project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_wsb_assembly_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        assembly_text = (output_dir / "assembly.py").read_text(encoding="utf-8")
        assert "from .ems import ems_comp as ems_static" in assembly_text
        module = importlib.import_module(
            f"hydesign.modular.projects.{project_name}.assembly"
        )
        assert module.__name__ == f"hydesign.modular.projects.{project_name}.assembly"
    finally:
        _drop_generated_modules(project_name)
        shutil.rmtree(output_dir, ignore_errors=True)
