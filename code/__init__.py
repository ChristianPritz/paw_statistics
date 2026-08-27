"""Paw posture analysis, visualization, and annotation tools.

The repository's ``code`` directory is installed as the
:mod:`paw_statistics` package. Submodules are loaded lazily so importing the
package itself does not initialize GUI or machine-learning dependencies.
"""

from importlib import import_module

__version__ = "1.0.0"

_MODULES = {
    "DataFrameViewerUI",
    "ImageSequenceExporter",
    "data_viewer",
    "interactive_plot_UI",
    "paw_statistics",
    "paw_UI",
    "plotter_UI",
    "weighted_keypoint_losses",
}

_OBJECTS = {
    "paw_statistics": ("paw_statistics", "paw_statistics"),
    "ImageSequenceExporter": ("paw_UI", "ImageSequenceExporter"),
    "ImageSequenceExporter2": ("paw_UI", "ImageSequenceExporter2"),
    "DataFrameViewerUI": ("DataFrameViewerUI", "DataFrameViewerUI"),
    "interactive_plot_UI": ("interactive_plot_UI", "interactive_plot_UI"),
    "PlotterUI": ("plotter_UI", "PlotterUI"),
}

__all__ = sorted(_MODULES | set(_OBJECTS))


def __getattr__(name):
    """Load public modules and objects only when first requested."""
    if name in _OBJECTS:
        module_name, object_name = _OBJECTS[name]
        value = getattr(import_module(f".{module_name}", __name__), object_name)
    elif name in _MODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
