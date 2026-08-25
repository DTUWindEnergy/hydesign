"""User-facing settings for contracted PPA delivery plots."""

# Palette definitions are maintained in post_processing/common/plotting_style.py.
# They are repeated here as a visible reference when selecting a palette:
#
# colorblind  = blue, green, grey, vermilion, purple
#               #0072B2, #009E73, #777777, #D55E00, #CC79A7
# colorblind2 = blue, green, brown, vermilion, purple
#               #0072B2, #009E73, #9A6A2F, #D55E00, #CC79A7
# colorblind3 = blue, green, brown, charcoal, mauve
#               #0072B2, #009E73, #9A6A2F, #666666, #B279A2
# academic    = dark blue, muted green, grey, brown, muted purple
#               #1F4E79, #5B8C5A, #7F7F7F, #9A6A2F, #6B5B95
# muted       = muted blue, teal, grey, orange, mauve
#               #4C78A8, #72B7B2, #A0A0A0, #F58518, #B279A2
AVAILABLE_PALETTES = (
    "colorblind",
    "colorblind2",
    "colorblind3",
    "academic",
    "muted",
)

PALETTE_NAME = "colorblind3"
