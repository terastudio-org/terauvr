"""
Enhanced configuration management system.

This module provides robust configuration loading, validation, and management
with support for multiple configuration formats and automatic device detection.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from ..utils.device_manager import get_device_manager, DeviceType


@dataclass
class AppConfig:
    """Application configuration data structure."""
    # Application settings
    app_port: int = 7860
    server_name: str = "0.0.0.0"
    num_of_restart: int = 5
    app_show_error: bool = False
    language: str = "vi-VN"
    theme: str = "NoCrypt/miku"
    
    # Device settings
    device: str = "auto"
    fp16: bool = False
    cpu_mode: bool = False
    brain_mode: bool = False
    
    # Performance settings
    gpu_memory_threshold: float = 0.8
    batch_size: int = 1
    num_workers: int = 4
    
    # Paths
    weights_path: str = "assets/models/uvr5"
    logs_path: str = "logs"
    audios_path: str = "audios"
    reference_path: str = "reference"
    presets_path: str = "presets"
    language_path: str = "assets/languages"
    
    # Advanced settings
    debug_mode: bool = False
    discord_presence: bool = True
    edge_tts: List[str] = None
    google_tts_voice: List[str] = None
    
    def __post_init__(self):
        """Initialize default values that depend on other fields."""
        if self.edge_tts is None:
            self.edge_tts = ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]
        if self.google_tts_voice is None:
            self.google_tts_voice = ["vi", "en"]


class ConfigManager:
    """
    Enhanced configuration manager with automatic device detection
    and robust error handling.
    """
    
    DEFAULT_CONFIG_PATHS = [
        "config.yaml",
        "config.yml", 
        "config.json",
        "configs/config.yaml",
        "configs/config.yml",
        "configs/config.json",
        "~/.terauvr/config.yaml",
        "~/.config/terauvr/config.yaml"
    ]
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._config_path = config_path
        self._config: AppConfig = AppConfig()
        self._translations: Dict[str, Any] = {}
        self._load_path = None
        
        # Load configuration
        self._load_configuration()
        
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in search paths."""
        # Check explicit path first
        if self._config_path:
            path = Path(self._config_path)
            if path.exists():
                return path
        
        # Search default paths
        for config_path in self.DEFAULT_CONFIG_PATHS:
            path = Path(config_path).expanduser()
            if path.exists():
                return path
        
        return None
    
    def _load_configuration(self):
        """Load configuration from file or create default."""
        config_file = self._find_config_file()
        
        if config_file:
            self.logger.info(f"Loading configuration from: {config_file}")
            self._load_path = config_file
            self._load_from_file(config_file)
        else:
            self.logger.info("No configuration file found, using defaults")
            self._config_path = None
            
        # Load translations
        self._load_translations()
        
        # Auto-detect device if needed
        self._setup_device_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
    def _load_from_file(self, config_file: Path):
        """Load configuration from a file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_ext = config_file.suffix.lower()
                
                if file_ext in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            # Update configuration with loaded values
            if isinstance(file_config, dict):
                for key, value in file_config.items():
                    if hasattr(self._config, key):
                        setattr(self._config, key, value)
                        
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise
    
    def _load_translations(self):
        """Load language translations."""
        try:
            # Find language files
            lang_dir = Path(self._config.language_path)
            
            if not lang_dir.exists():
                self.logger.warning(f"Language directory not found: {lang_dir}")
                return
            
            # Load primary language
            lang_file = lang_dir / f"{self._config.language}.json"
            if not lang_file.exists():
                # Try English as fallback
                lang_file = lang_dir / "en-US.json"
            
            if lang_file.exists():
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self._translations = json.load(f)
                self.logger.info(f"Loaded translations from: {lang_file}")
            else:
                self.logger.warning(f"No translation file found for {self._config.language}")
                
        except Exception as e:
            self.logger.warning(f"Failed to load translations: {e}")
            self._translations = {}
    
    def _setup_device_configuration(self):
        """Setup device configuration with auto-detection."""
        if self._config.device == "auto" or not self._config.device:
            # Auto-detect the best device
            device_manager = get_device_manager()
            best_device = device_manager.get_best_device()
            
            if best_device:
                device_map = {
                    DeviceType.CUDA: "cuda",
                    DeviceType.DIRECTML: "privateuseone",
                    DeviceType.OPENCL: "ocl", 
                    DeviceType.MPS: "mps",
                    DeviceType.CPU: "cpu"
                }
                
                self._config.device = device_map.get(best_device.device_type, "cpu")
                self.logger.info(f"Auto-detected device: {best_device.name} -> {self._config.device}")
            else:
                self._config.device = "cpu"
                self.logger.warning("No suitable device found, using CPU")
        
        # Adjust FP16 setting based on device capability
        self._configure_fp16()
    
    def _configure_fp16(self):
        """Configure FP16 settings based on device."""
        device = self._config.device
        
        # Disable FP16 on devices that don't support it well
        fp16_incompatible = ["cpu", "mps"]
        
        if device.startswith("ocl") or device.startswith("privateuseone"):
            # DirectML and OpenCL might support FP16 but it can be problematic
            if not self._config.fp16:
                # Don't enable FP16 automatically
                pass
            else:
                self.logger.warning("FP16 enabled with DirectML/OpenCL - may cause issues")
        elif device in fp16_incompatible:
            if self._config.fp16:
                self.logger.warning(f"FP16 requested but not compatible with {device}, disabling")
                self._config.fp16 = False
    
    def _validate_configuration(self):
        """Validate the loaded configuration."""
        # Validate port range
        if not (1024 <= self._config.app_port <= 65535):
            self.logger.warning(f"Port {self._config.app_port} is outside valid range, using default")
            self._config.app_port = 7860
        
        # Validate batch size
        if self._config.batch_size < 1:
            self.logger.warning("Invalid batch size, using default")
            self._config.batch_size = 1
        
        # Validate num_workers
        if self._config.num_workers < 1:
            self.logger.warning("Invalid num_workers, using default")
            self._config.num_workers = 4
        
        # Validate paths
        for path_attr in ['weights_path', 'logs_path', 'audios_path', 'reference_path', 'presets_path']:
            path = getattr(self._config, path_attr)
            if not Path(path).is_absolute():
                # Convert relative paths to absolute
                setattr(self._config, path_attr, str(Path(path).resolve()))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self._config, key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        if hasattr(self._config, key):
            setattr(self._config, key, value)
        else:
            self.logger.warning(f"Unknown configuration key: {key}")
    
    def get_translations(self) -> Dict[str, Any]:
        """Get loaded translations."""
        return self._translations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self._config)
    
    def save(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        if output_path is None:
            output_path = self._load_path if self._load_path else "config.yaml"
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Configuration saved to: {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {path}: {e}")
            raise
    
    @contextmanager
    def temporary_config(self, **kwargs):
        """Context manager for temporary configuration changes."""
        # Store original values
        original_values = {}
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                original_values[key] = getattr(self._config, key)
                setattr(self._config, key, value)
        
        try:
            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(self._config, key, value)
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("TERAUVR - Configuration Summary")
        print("="*60)
        print(f"Config File: {self._load_path or 'Using defaults'}")
        print(f"Device: {self._config.device}")
        print(f"Language: {self._config.language}")
        print(f"Theme: {self._config.theme}")
        print(f"Port: {self._config.app_port}")
        print(f"Server: {self._config.server_name}")
        print(f"FP16: {self._config.fp16}")
        print(f"Debug Mode: {self._config.debug_mode}")
        print()
        print("Paths:")
        print(f"  Weights: {self._config.weights_path}")
        print(f"  Logs: {self._config.logs_path}")
        print(f"  Audios: {self._config.audios_path}")
        print()
        if self._translations:
            print(f"Translations: Loaded ({len(self._translations)} keys)")
        else:
            print("Translations: Not loaded")
        print("="*60)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Convenience function to get configuration value."""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any):
    """Convenience function to set configuration value."""
    get_config_manager().set(key, value)