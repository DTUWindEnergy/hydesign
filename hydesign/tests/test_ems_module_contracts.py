from hydesign.modules.ems import AVAILABLE_EMS_MODULES, compose_ems_modules


def test_compose_ems_modules_adds_base_and_preserves_order():
    modules = compose_ems_modules(["Wind", "solar", "battery"])

    assert [module.name for module in modules] == ["base", "wind", "solar", "battery"]


def test_compose_ems_modules_removes_duplicates():
    modules = compose_ems_modules(["wind", "wind", "solar", "battery", "solar"])

    assert [module.name for module in modules] == ["base", "wind", "solar", "battery"]


def test_available_ems_modules_are_grounded_in_original_ems_sources():
    for name, module in AVAILABLE_EMS_MODULES.items():
        assert module.source_of_truth
        assert all(path.startswith("hydesign/ems/") for path in module.source_of_truth)
        assert module.name == name
