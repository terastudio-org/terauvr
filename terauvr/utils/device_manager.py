"""
Enhanced device detection and management system.

This module provides robust automatic device detection and management
for GPU acceleration in audio processing tasks.
"""

import os
import sys
import logging
import platform
import subprocess
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import torch


class DeviceType(Enum):
    """Enumeration of supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    OPENCL = "opencl"
    MPS = "mps"


@dataclass
class DeviceInfo:
    """Information about a detected device."""
    device_type: DeviceType
    device_id: int
    name: str
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    is_available: bool = False
    driver_version: Optional[str] = None


class DeviceManager:
    """
    Advanced device detection and management system.
    
    This class provides comprehensive device detection with automatic
    fallback mechanisms and health checking capabilities.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the device manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._detected_devices: List[DeviceInfo] = []
        self._preferred_order = [
            DeviceType.CUDA,
            DeviceType.DIRECTML, 
            DeviceType.OPENCL,
            DeviceType.MPS,
            DeviceType.CPU
        ]
        
    def detect_all_devices(self) -> List[DeviceInfo]:
        """Detect all available devices on the system."""
        self.logger.info("Scanning for available compute devices...")
        self._detected_devices = []
        
        # Detect CUDA devices
        cuda_devices = self._detect_cuda_devices()
        self._detected_devices.extend(cuda_devices)
        
        # Detect DirectML devices
        directml_devices = self._detect_directml_devices()
        self._detected_devices.extend(directml_devices)
        
        # Detect OpenCL devices
        opencl_devices = self._detect_opencl_devices()
        self._detected_devices.extend(opencl_devices)
        
        # Detect Metal Performance Shaders (MPS)
        mps_devices = self._detect_mps_devices()
        self._detected_devices.extend(mps_devices)
        
        # Always include CPU as fallback
        cpu_device = DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name="CPU (x86_64)",
            memory_gb=None,
            is_available=True
        )
        self._detected_devices.append(cpu_device)
        
        self.logger.info(f"Detected {len(self._detected_devices)} devices")
        for device in self._detected_devices:
            status = "Available" if device.is_available else "Not Available"
            self.logger.info(f"  {device.device_type.value}:{device.device_id} - {device.name} ({status})")
        
        return self._detected_devices
    
    def get_recommended_device(self, prefer_gpu: bool = True) -> Optional[DeviceInfo]:
        """
        Get the recommended device for computation.
        
        Args:
            prefer_gpu: Whether to prefer GPU acceleration over CPU
            
        Returns:
            DeviceInfo object for the recommended device, or None if no suitable device found
        """
        devices = self.detect_all_devices()
        
        if not prefer_gpu:
            return self._detected_devices[0]  # Return CPU as fallback
        
        # Try devices in preferred order
        for device_type in self._preferred_order:
            if device_type == DeviceType.CPU and not prefer_gpu:
                continue
                
            for device in devices:
                if device.device_type == device_type and device.is_available:
                    self.logger.info(f"Recommended device: {device.name}")
                    return device
        
        # Fallback to CPU
        self.logger.warning("No suitable GPU device found, falling back to CPU")
        return devices[-1]  # Should be CPU
    
    def get_best_device(self) -> Optional[DeviceInfo]:
        """
        Get the best available device based on performance capabilities.
        
        Returns:
            DeviceInfo for the best available device
        """
        devices = self.detect_all_devices()
        
        # Filter available devices and sort by preference
        available_devices = [d for d in devices if d.is_available]
        
        # Score devices based on their capabilities
        def score_device(device: DeviceInfo) -> int:
            scores = {
                DeviceType.CUDA: 1000,
                DeviceType.DIRECTML: 800,
                DeviceType.OPENCL: 700,
                DeviceType.MPS: 600,
                DeviceType.CPU: 100
            }
            
            base_score = scores.get(device.device_type, 0)
            
            # Add memory bonus
            if device.memory_gb:
                if device.memory_gb >= 16:
                    base_score += 200
                elif device.memory_gb >= 8:
                    base_score += 100
                elif device.memory_gb >= 4:
                    base_score += 50
            
            # Add compute capability bonus for CUDA
            if device.device_type == DeviceType.CUDA and device.compute_capability:
                try:
                    cc_major, cc_minor = map(float, device.compute_capability.split('.'))
                    base_score += int(cc_major * 100 + cc_minor * 10)
                except (ValueError, AttributeError):
                    pass
            
            return base_score
        
        available_devices.sort(key=score_device, reverse=True)
        
        if available_devices:
            best_device = available_devices[0]
            self.logger.info(f"Best device selected: {best_device.name}")
            return best_device
        
        return None
    
    def health_check(self, device_info: DeviceInfo) -> bool:
        """
        Perform a health check on a specific device.
        
        Args:
            device_info: Device information to check
            
        Returns:
            True if device passes health check, False otherwise
        """
        try:
            if device_info.device_type == DeviceType.CUDA:
                if not torch.cuda.is_available():
                    return False
                
                device = torch.device(f"cuda:{device_info.device_id}")
                test_tensor = torch.randn(100, 100).to(device)
                result = test_tensor @ test_tensor
                return result.device.type == f"cuda:{device_info.device_id}"
                
            elif device_info.device_type == DeviceType.MPS:
                if not torch.backends.mps.is_available():
                    return False
                
                device = torch.device("mps")
                test_tensor = torch.randn(100, 100).to(device)
                result = test_tensor @ test_tensor
                return result.device.type == "mps"
                
            elif device_info.device_type in [DeviceType.DIRECTML, DeviceType.OPENCL]:
                # Basic availability check
                return device_info.is_available
                
            elif device_info.device_type == DeviceType.CPU:
                # CPU is always available
                return True
                
        except Exception as e:
            self.logger.warning(f"Health check failed for {device_info.name}: {e}")
            return False
        
        return True
    
    def _detect_cuda_devices(self) -> List[DeviceInfo]:
        """Detect CUDA-capable devices."""
        devices = []
        
        try:
            if not torch.cuda.is_available():
                return devices
                
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                try:
                    device_id = i
                    device_name = torch.cuda.get_device_name(i)
                    memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    compute_capability = f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}"
                    
                    device = DeviceInfo(
                        device_type=DeviceType.CUDA,
                        device_id=device_id,
                        name=device_name,
                        memory_gb=memory_gb,
                        compute_capability=compute_capability,
                        is_available=True
                    )
                    devices.append(device)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get CUDA device {i} info: {e}")
                    
        except Exception as e:
            self.logger.warning(f"CUDA detection failed: {e}")
            
        return devices
    
    def _detect_directml_devices(self) -> List[DeviceInfo]:
        """Detect DirectML devices."""
        devices = []
        
        try:
            # Check if torch-directml is available
            torch_directml = self._try_import_directml()
            if not torch_directml:
                return devices
                
            if torch_directml.is_available():
                device_count = torch_directml.device_count()
                
                for i in range(device_count):
                    try:
                        device_name = torch_directml.device_name(i)
                        
                        device = DeviceInfo(
                            device_type=DeviceType.DIRECTML,
                            device_id=i,
                            name=device_name,
                            memory_gb=None,  # DirectML doesn't expose memory info easily
                            is_available=True
                        )
                        devices.append(device)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get DirectML device {i} info: {e}")
                        
        except Exception as e:
            self.logger.warning(f"DirectML detection failed: {e}")
            
        return devices
    
    def _detect_opencl_devices(self) -> List[DeviceInfo]:
        """Detect OpenCL devices."""
        devices = []
        
        try:
            # Check if pytorch-ocl is available
            pytorch_ocl = self._try_import_ocl()
            if not pytorch_ocl:
                return devices
                
            # OpenCL device detection is platform-specific
            gpu_list = self._get_opencl_gpu_list()
            
            for i, gpu_name in enumerate(gpu_list):
                device = DeviceInfo(
                    device_type=DeviceType.OPENCL,
                    device_id=i,
                    name=gpu_name,
                    memory_gb=None,
                    is_available=True
                )
                devices.append(device)
                
        except Exception as e:
            self.logger.warning(f"OpenCL detection failed: {e}")
            
        return devices
    
    def _detect_mps_devices(self) -> List[DeviceInfo]:
        """Detect Metal Performance Shaders (MPS) devices."""
        devices = []
        
        try:
            if torch.backends.mps.is_available():
                device = DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_id=0,
                    name="Apple Silicon GPU (MPS)",
                    memory_gb=None,
                    is_available=True
                )
                devices.append(device)
                
        except Exception as e:
            self.logger.warning(f"MPS detection failed: {e}")
            
        return devices
    
    def _try_import_directml(self):
        """Try to import torch_directml."""
        try:
            import torch_directml
            return torch_directml
        except ImportError:
            return None
    
    def _try_import_ocl(self):
        """Try to import pytorch_ocl."""
        try:
            import pytorch_ocl
            return pytorch_ocl
        except ImportError:
            return None
    
    def _get_opencl_gpu_list(self) -> List[str]:
        """Get list of OpenCL-capable GPU devices."""
        gpu_list = []
        
        try:
            if platform.system() == "Windows":
                gpu_list = self._get_amd_gpu_windows()
            elif platform.system() == "Linux":
                gpu_list = self._get_amd_gpu_linux()
                
        except Exception as e:
            self.logger.warning(f"Failed to detect OpenCL GPUs: {e}")
            
        return gpu_list
    
    def _check_amd_gpu(self, gpu_name: str) -> bool:
        """Check if GPU name contains AMD/Vega indicators."""
        amd_indicators = ["RX", "AMD", "Vega", "Radeon", "FirePro"]
        return any(indicator in gpu_name for indicator in amd_indicators)
    
    def _get_amd_gpu_windows(self) -> List[str]:
        """Get AMD GPU list on Windows."""
        gpus = []
        
        try:
            # Try using WMIC first
            try:
                output = subprocess.check_output(
                    "wmic path win32_VideoController get name",
                    shell=True,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError:
                # Fallback to PowerShell
                output = subprocess.check_output(
                    'powershell "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"',
                    shell=True,
                    stderr=subprocess.DEVNULL
                )
            
            lines = output.decode().split('\n')[1:]  # Skip header
            gpus = [gpu.strip() for gpu in lines if self._check_amd_gpu(gpu)]
            
        except Exception as e:
            self.logger.warning(f"Failed to get Windows GPU list: {e}")
            
        return gpus
    
    def _get_amd_gpu_linux(self) -> List[str]:
        """Get AMD GPU list on Linux."""
        gpus = []
        
        try:
            output = subprocess.check_output(
                "lspci | grep VGA",
                shell=True
            )
            
            lines = output.decode().split('\n')
            gpus = [gpu for gpu in lines if self._check_amd_gpu(gpu)]
            
        except Exception as e:
            self.logger.warning(f"Failed to get Linux GPU list: {e}")
            
        return gpus
    
    def get_device_count(self) -> int:
        """Get the total number of detected devices."""
        return len(self._detected_devices)
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """Get only the devices that are available for use."""
        return [d for d in self._detected_devices if d.is_available]
    
    def format_device_info(self, device_info: DeviceInfo) -> str:
        """Format device information for display."""
        memory_str = f", {device_info.memory_gb:.1f}GB" if device_info.memory_gb else ""
        cc_str = f", {device_info.compute_capability}" if device_info.compute_capability else ""
        status_str = "Available" if device_info.is_available else "Not Available"
        
        return f"{device_info.device_type.value.upper()}:{device_info.device_id} - {device_info.name}{memory_str}{cc_str} ({status_str})"
    
    def print_device_summary(self):
        """Print a summary of all detected devices."""
        print("\n" + "="*80)
        print("TERAUVR - Device Detection Summary")
        print("="*80)
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        print()
        print("Detected Devices:")
        print("-" * 80)
        
        for device in self._detected_devices:
            print(f"  {self.format_device_info(device)}")
        
        print()
        recommended = self.get_recommended_device()
        best = self.get_best_device()
        
        if recommended:
            print(f"Recommended Device: {recommended.name}")
        if best and best.device_id != recommended.device_id if recommended else True:
            print(f"Best Performance: {best.name}")
        
        print("="*80)


# Global device manager instance
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    return device_manager


def detect_devices() -> List[DeviceInfo]:
    """Convenience function to detect all available devices."""
    return device_manager.detect_all_devices()


def get_recommended_device(prefer_gpu: bool = True) -> Optional[DeviceInfo]:
    """Convenience function to get recommended device."""
    return device_manager.get_recommended_device(prefer_gpu)


def get_best_device() -> Optional[DeviceInfo]:
    """Convenience function to get best available device."""
    return device_manager.get_best_device()