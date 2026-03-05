import json
import shutil
from pathlib import Path

from hydesign.project_builder import (
    BuildManifest,
    BuildRequest,
    build_baseline_cplex_ems_project,
    build_baseline_wsb_costs_project,
    build_baseline_wsb_finance_project,
    should_build,
)


def test_build_request_normalizes_names_and_paths():
    request = BuildRequest(
        project_name=" Demo Project ",
        technologies=["Wind", "solar", "wind", "Battery"],
        artifacts=["EMS", "assembly", "ems"],
    )

    assert request.project_name == "Demo Project"
    assert request.project_slug == "demo_project"
    assert request.technologies == ("wind", "solar", "battery")
    assert request.artifacts == ("ems", "assembly")
    assert request.output_dir == Path("hydesign") / "combined_projects" / "demo_project"
    assert request.manifest_path == request.output_dir / "manifest.yml"


def test_build_request_can_derive_project_name_from_technologies():
    request = BuildRequest(
        project_name="",
        technologies=["Wind", "solar"],
        artifacts=["assembly"],
    )

    assert request.project_name == "wind_solar"
    assert request.project_slug == "wind_solar"


def test_should_build_respects_toggle_and_manifest_matching():
    request = BuildRequest(
        project_name="baseline",
        technologies=["wind", "solar", "battery"],
        artifacts=["ems", "finance", "assembly"],
    )
    manifest = BuildManifest.from_request(request)

    assert should_build(request, manifest) is False
    assert (
        should_build(
            BuildRequest(
                project_name="baseline",
                technologies=["wind", "solar", "battery", "p2x"],
                artifacts=["ems", "finance", "assembly"],
            ),
            manifest,
        )
        is True
    )
    assert (
        should_build(
            BuildRequest(
                project_name="baseline",
                technologies=["wind", "solar", "battery"],
                artifacts=["ems", "finance", "assembly"],
                enable_build=False,
            ),
            manifest,
        )
        is False
    )
    assert (
        should_build(
            BuildRequest(
                project_name="baseline",
                technologies=["wind", "solar", "battery"],
                artifacts=["ems", "finance", "assembly"],
                regenerate=True,
            ),
            manifest,
        )
        is True
    )


def test_sequential_wsb_build_prunes_stale_source_hashes():
    project_name = "pytest_manifest_hash_pruning"
    output_dir = (
        Path.cwd() / "hydesign" / "combined_projects" / project_name
    ).resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_cplex_ems_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        build_baseline_wsb_costs_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        build_baseline_wsb_finance_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )

        manifest = json.loads((output_dir / "manifest.yml").read_text(encoding="utf-8"))
        assert set(manifest["source_hashes"]) == {
            "hydesign/ems/ems.py",
            "hydesign/costs/costs.py",
            "hydesign/finance/finance.py",
        }
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
