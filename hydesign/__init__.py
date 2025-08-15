try:  # pragma: no cover
    from hydesign._version import __version__

    __release__ = __version__
except BaseException:  # pragma: no cover
    pass
