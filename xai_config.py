DEFAULT_XAI_METHODS = ["Grad-CAM", "LIME", "SHAP"]

XAI_METHODS_BY_EXTENSION = {
    ".png": ["Grad-CAM", "LIME", "SHAP"],
    ".jpg": ["Grad-CAM", "LIME", "SHAP"],
    ".jpeg": ["Grad-CAM", "LIME", "SHAP"],
    ".wav": ["Grad-CAM", "LIME", "SHAP"],
}


def get_xai_methods(file_ext: str) -> list[str]:
    """Return the configured XAI methods for a given file extension."""

    methods = XAI_METHODS_BY_EXTENSION.get(file_ext, DEFAULT_XAI_METHODS)
    return list(methods)
