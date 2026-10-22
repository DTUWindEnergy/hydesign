"""Shared color palettes for thesis figures."""


PALETTES = {
    "colorblind": ["#0072B2", "#009E73", "#777777", "#C05503", "#CC79A7"],
    "colorblind2": ["#0072B2", "#009E73", "#795426", "#C05503", "#B279A2"],
    "colorblind3": ["#0072B2",  "#009E73",  "#9A6A2F", "#666666",  "#B279A2"],
    "colorblind4": ["#0072B2",  "#009E73",  "#666666",  "#9A6A2F",  "#B279A2"],
    "colorblind5": ["#0072B2",  "#009E73",  "#666666",  "#6B5B95",  "#9A6A2F"],
    "academic": ["#1F4E79", "#5B8C5A", "#7F7F7F", "#9A6A2F", "#6B5B95"],
    "muted": ["#4C78A8", "#72B7B2", "#A0A0A0", "#F58518", "#B279A2"],
}


def get_palette(name: str) -> list[str]:
    try:
        return PALETTES[name]
    except KeyError as exc:
        available = ", ".join(PALETTES)
        raise ValueError(
            f"Unknown color palette '{name}'. Choose one of: {available}."
        ) from exc
