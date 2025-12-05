"""
Utility modules for the terauvr package.

This module provides various utility functions and classes for device management,
logging, audio processing, and other common operations.
"""

from .device_manager import (
    DeviceManager,
    DeviceInfo,
    DeviceType,
    get_device_manager,
    detect_devices,
    get_recommended_device,
    get_best_device,
)

__all__ = [
    "DeviceManager",
    "DeviceInfo", 
    "DeviceType",
    "get_device_manager",
    "detect_devices",
    "get_recommended_device", 
    "get_best_device",
]