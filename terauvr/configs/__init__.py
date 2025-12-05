"""
Configuration management module.

This module provides configuration loading, validation, and management
with support for multiple configuration formats and automatic device detection.
"""

from .config import (
    ConfigManager,
    AppConfig,
    get_config_manager,
    get_config,
    set_config,
)

__all__ = [
    "ConfigManager",
    "AppConfig",
    "get_config_manager",
    "get_config", 
    "set_config",
]