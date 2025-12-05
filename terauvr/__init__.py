"""
TeraStudio UVR - Advanced AI-powered VR/AR audio separation toolkit.

This package provides state-of-the-art audio source separation using various
deep learning models including VR, Demucs, and MDX architectures.

Author: terastudio-org/terastudio
License: MIT
"""

__version__ = "1.0.0"
__author__ = "terastudio-org/terastudio"
__email__ = "contact@terastudio.org"
__license__ = "MIT"
__homepage__ = "https://github.com/terastudio-org/terauvr"

# Package metadata
__all__ = [
    "__version__",
    "__author__", 
    "__license__",
    "__homepage__",
    "app",
    "library", 
    "configs",
    "utils",
]

# Import main components
try:
    from .utils.device_manager import DeviceManager
    from .configs.config import Config
    from .app import create_app
    
    __all__.extend([
        "DeviceManager",
        "Config", 
        "create_app"
    ])
except ImportError as e:
    # Graceful handling of missing dependencies during installation
    import warnings
    warnings.warn(
        f"Some dependencies may be missing: {e}. "
        "Install with 'pip install terauvr[all]' for full functionality.",
        ImportWarning
    )

# Package initialization
def get_version():
    """Get the current version of terauvr."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "terauvr",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "homepage": __homepage__,
    }
