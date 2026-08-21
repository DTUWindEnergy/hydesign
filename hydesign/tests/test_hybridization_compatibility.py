import pytest

from hydesign.assembly import (
    hpp_assembly_hybridization_pv,
    hpp_assembly_hybridization_wind,
)


@pytest.mark.parametrize(
    "legacy_module",
    [hpp_assembly_hybridization_pv, hpp_assembly_hybridization_wind],
)
def test_legacy_hybridization_models_warn(monkeypatch, legacy_module):
    monkeypatch.setattr(legacy_module._base_hpp_model, "__init__", lambda self: None)

    with pytest.warns(DeprecationWarning, match="operation-year inputs"):
        legacy_module.hpp_model()
