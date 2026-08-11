"""Technology deployment schedule parsing and utilities."""

from collections.abc import Mapping, Sequence

TECHNOLOGY_ALIASES = {
    "wind": "wind",
    "wpp": "wind",
    "solar": "solar",
    "pv": "solar",
    "pvp": "solar",
    "battery": "battery",
    "bess": "battery",
    "storage": "battery",
}
SUPPORTED_TECHNOLOGIES = ("wind", "solar", "battery")


def normalize_technology_schedule(technologies, default_lifetime=None):
    """Validate and normalize a technology deployment configuration.

    ``technologies`` may be a list of mappings containing ``name``,
    ``deployment_year`` and ``lifetime``, or a mapping keyed by technology
    name. Common wind, PV and storage aliases are accepted.
    """
    if technologies is None:
        return None

    if isinstance(technologies, Mapping):
        entries = []
        for name, values in technologies.items():
            if values is None:
                values = {}
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"Technology '{name}' must contain deployment_year and lifetime."
                )
            entries.append(dict(values, name=name))
    elif isinstance(technologies, Sequence) and not isinstance(technologies, str):
        entries = list(technologies)
    else:
        raise ValueError("'technologies' must be a list or mapping.")

    if not entries:
        raise ValueError("'technologies' must contain at least one technology.")

    schedule = {}
    for entry in entries:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, Mapping) or "name" not in entry:
            raise ValueError("Each technology must be a mapping with a 'name'.")

        supplied_name = str(entry["name"]).lower()
        try:
            name = TECHNOLOGY_ALIASES[supplied_name]
        except KeyError as exc:
            supported = ", ".join(SUPPORTED_TECHNOLOGIES)
            raise ValueError(
                f"Unsupported technology '{supplied_name}'. Supported: {supported}."
            ) from exc
        if name in schedule:
            raise ValueError(f"Technology '{name}' is specified more than once.")

        deployment_year = entry.get(
            "deployment_year", entry.get("year", entry.get("start_year", 0))
        )
        lifetime = entry.get("lifetime", entry.get("life_y", default_lifetime))
        if lifetime is None:
            raise ValueError(
                f"Technology '{name}' requires a lifetime when no project life_y is set."
            )
        if not _is_integer(deployment_year) or int(deployment_year) < 0:
            raise ValueError(
                f"Technology '{name}' deployment_year must be a non-negative integer."
            )
        if not _is_integer(lifetime) or int(lifetime) <= 0:
            raise ValueError(
                f"Technology '{name}' lifetime must be a positive integer."
            )

        deployment_year = int(deployment_year)
        lifetime = int(lifetime)
        capex_phasing = _normalize_capex_phasing(entry, name)
        schedule[name] = {
            "deployment_year": deployment_year,
            "lifetime": lifetime,
            "end_year": deployment_year + lifetime,
            "capex_phasing": capex_phasing,
        }
    return schedule


def legacy_technology_schedule(
    life_y,
    wind_operation_year=0,
    solar_operation_year=0,
    battery_operation_year=0,
):
    """Convert the legacy per-technology start years to a schedule."""
    return {
        name: {
            "deployment_year": int(start_year),
            "lifetime": int(life_y),
            "end_year": int(start_year + life_y),
        }
        for name, start_year in (
            ("wind", wind_operation_year),
            ("solar", solar_operation_year),
            ("battery", battery_operation_year),
        )
    }


def project_lifetime(schedule):
    """Return the calendar horizon required by a normalized schedule."""
    return max(item["end_year"] for item in schedule.values())


def technology_is_active(schedule, technology, year):
    """Return whether a technology is operating during a project year."""
    item = schedule.get(technology)
    return item is not None and item["deployment_year"] <= year < item["end_year"]


def technology_stage_intervals(schedule, project_life_y=None):
    """Return contiguous project intervals with an unchanged technology set."""
    if project_life_y is None:
        project_life_y = project_lifetime(schedule)

    def active_technologies(year):
        return tuple(
            name
            for name in SUPPORTED_TECHNOLOGIES
            if technology_is_active(schedule, name, year)
        )

    stages = []
    start_year = 0
    active = active_technologies(0)
    for year in range(1, project_life_y):
        next_active = active_technologies(year)
        if next_active != active:
            stages.append(
                {
                    "start_year": start_year,
                    "end_year": year,
                    "technologies": active,
                }
            )
            start_year = year
            active = next_active
    stages.append(
        {
            "start_year": start_year,
            "end_year": project_life_y,
            "technologies": active,
        }
    )
    return stages


def _is_integer(value):
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _normalize_capex_phasing(entry, technology):
    """Return normalized CAPEX payment years relative to deployment."""
    phasing = entry.get("capex_phasing")
    if phasing is None and ("phasing_yr" in entry or "phasing_CAPEX" in entry):
        phasing = {
            "years": entry.get("phasing_yr"),
            "shares": entry.get("phasing_CAPEX"),
        }
    if phasing is None:
        return {"years": [0], "shares": [1.0]}
    if not isinstance(phasing, Mapping):
        raise ValueError(f"Technology '{technology}' capex_phasing must be a mapping.")

    years = phasing.get("years", phasing.get("phasing_yr"))
    shares = phasing.get("shares", phasing.get("phasing_CAPEX"))
    if years is None or shares is None:
        raise ValueError(
            f"Technology '{technology}' capex_phasing requires years and shares."
        )
    if (
        not isinstance(years, Sequence)
        or isinstance(years, str)
        or not isinstance(shares, Sequence)
        or isinstance(shares, str)
        or len(years) != len(shares)
        or not years
    ):
        raise ValueError(
            f"Technology '{technology}' capex_phasing years and shares must be equally sized, non-empty lists."
        )
    if any(not _is_integer(year) or int(year) > 0 for year in years):
        raise ValueError(
            f"Technology '{technology}' CAPEX phasing years must be integer years at or before deployment."
        )
    try:
        shares = [float(share) for share in shares]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Technology '{technology}' CAPEX phasing shares must be numeric."
        ) from exc
    if any(not 0 <= share for share in shares) or sum(shares) <= 0:
        raise ValueError(
            f"Technology '{technology}' CAPEX phasing shares must be non-negative with a positive sum."
        )

    total = sum(shares)
    return {
        "years": [int(year) for year in years],
        "shares": [share / total for share in shares],
    }
