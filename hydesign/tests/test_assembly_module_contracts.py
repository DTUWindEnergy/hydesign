from hydesign.modules.assembly import compose_supported_assembly_modules


def test_wsb_assembly_registry_returns_expected_module_order():
    modules = compose_supported_assembly_modules(("wind", "solar", "battery"))
    assert [module.name for module in modules] == [
        "base",
        "wind",
        "solar",
        "battery",
        "operations",
        "finance",
    ]
