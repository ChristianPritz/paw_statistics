"""Public interface for the paw statistics package."""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "paw_statistics": ("paw_statistics", "paw_statistics"),
    "interactive_plot_UI": ("interactive_plot_UI", "interactive_plot_UI"),
    "PlotterUI": ("plotter_UI", "PlotterUI"),
    "ImageSequenceExporter": ("paw_UI", "ImageSequenceExporter"),
    "ImageSequenceExporter2": ("paw_UI", "ImageSequenceExporter2"),
    "ObjectDetection": ("paw_UI", "ObjectDetection"),
    "PawClass": ("paw_UI", "PawClass"),
    "PawDetection": ("paw_UI", "PawDetection"),
    "paw_cropper": ("paw_UI", "paw_cropper"),
    "paw_detector": ("paw_UI", "paw_detector"),
    "DataFrameViewerUI": ("DataFrameViewerUI", "DataFrameViewerUI"),
    # Compatibility spelling for callers that use a lowercase "f".
    "DataframeViewerUI": ("DataFrameViewerUI", "DataFrameViewerUI"),
    "DynamicWeightedOKSLoss": ("weighted_keypoint_losses", "DynamicWeightedOKSLoss"),
    "WeightedOKSKeypointLoss": ("weighted_keypoint_losses", "WeightedOKSKeypointLoss"),
    "WeightedPoseLoss": ("weighted_keypoint_losses", "WeightedPoseLoss"),
    "WeightedPoseModel": ("weighted_keypoint_losses", "WeightedPoseModel"),
    "WeightedSmoothL1KeypointLoss": ("weighted_keypoint_losses", "WeightedSmoothL1KeypointLoss"),
    "make_weighted_pose_trainer": ("weighted_keypoint_losses", "make_weighted_pose_trainer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load a public object without importing every optional UI dependency."""
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), object_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
