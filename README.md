# TeraStudio UVR

Advanced AI-powered VR/AR audio separation and processing toolkit.

## Overview

TeraStudio UVR is a comprehensive package for audio source separation using state-of-the-art deep learning models. It provides automatic device detection and optimization for various compute backends including CUDA, DirectML, OpenCL, and Apple Silicon (MPS).

**Author**: terastudio-org/terastudio  
**License**: MIT  
**Version**: 1.0.0

## Features

- 🎯 **Multi-stem audio separation** - Extract vocals, drums, bass, instruments
- 💻 **Automatic device detection** - Smart GPU acceleration with fallback
- 🔥 **Multiple compute backends** - CUDA, DirectML, OpenCL, MPS, CPU
- 🎨 **Modern web interface** - Built with Gradio for easy use
- 🌐 **Multi-language support** - English and Vietnamese
- ⚙️ **Flexible configuration** - YAML/JSON configuration support
- 🚀 **Performance optimization** - Automatic batch size and memory management

## Installation

### Basic Installation
```bash
pip install terauvr
```

### Full Installation (with GPU support)
```bash
pip install terauvr[all]
```

### Development Installation
```bash
git clone https://github.com/terastudio-org/terauvr.git
cd terauvr
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# Detect available devices
terauvr devices

# Show configuration
terauvr config show

# Launch GUI
terauvr gui

# Separate audio file
terauvr separate input.wav --output-dir ./output --model htdemucs
```

### Python API

```python
from terauvr import create_app, DeviceManager, ConfigManager

# Create and launch the GUI
app = create_app()
app.launch()

# Use the device manager
device_manager = DeviceManager()
best_device = device_manager.get_best_device()
print(f"Using device: {best_device.name}")

# Use the configuration manager
config_manager = ConfigManager()
config_manager.set('language', 'en-US')
```

### Library Usage

```python
import terauvr

# Get package information
info = terauvr.get_info()
print(f"Version: {info['version']}")

# Get recommended device
recommended_device = terauvr.get_recommended_device()
print(f"Recommended: {recommended_device.name}")
```

## Configuration

The package supports configuration files in YAML or JSON format:

### Example config.yaml
```yaml
device: auto
language: en-US
theme: gradio/soft
fp16: false
app_port: 7860
gpu_memory_threshold: 0.8
batch_size: 1
```

Configuration search paths:
- `./config.yaml`
- `./config.json`
- `./configs/config.yaml`
- `~/.terauvr/config.yaml`
- `~/.config/terauvr/config.yaml`

## Device Detection

The package automatically detects and configures the best available compute device:

1. **CUDA** (NVIDIA GPUs) - Best performance
2. **DirectML** (Windows AMD/Intel GPUs) - Good performance  
3. **OpenCL** (AMD GPUs) - Good performance
4. **MPS** (Apple Silicon) - Good performance
5. **CPU** - Always available fallback

## Supported Models

### MDX-Net Models
- UVR-MDX-NET Main (340, 390, 406, 427, 438)
- UVR-MDX-NET Instrumental variants
- Kim vocal/instrument models

### Demucs Models
- HT-Demucs
- HT-Demucs 6S
- HD-MMI variants

### VR Models  
- HP-UVR series
- SP-UVR series
- Various vocal enhancement models

### Specialized Models
- Karaoke separation
- Reverb/echo removal
- Noise reduction

## Command Reference

### `terauvr devices`
Detect and display available compute devices.

Options:
- `--prefer-gpu/--no-gpu` - Whether to prefer GPU acceleration
- `--format table|json` - Output format

### `terauvr config`
Manage configuration files.

Subcommands:
- `show [--format yaml|json]` - Display current configuration
- `generate [--output path] [--format yaml|json]` - Generate default config

### `terauvr separate`
Separate audio into individual tracks.

Arguments:
- `input_file` - Audio file to process

Options:
- `--output-dir, -o` - Output directory (default: ./output)
- `--model, -m` - Model to use (default: htdemucs)
- `--device, -d` - Compute device (default: auto)
- `--stems` - Comma-separated stems to extract

### `terauvr gui`
Launch the graphical user interface.

Options:
- `--port` - Server port (default: 7860)
- `--host` - Host to bind (default: 0.0.0.0)
- `--share` - Create public share link
- `--device` - Compute device (default: auto)
- `--config` - Configuration file path

## Architecture

### Package Structure
```
terauvr/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point
├── app/                 # Application modules
│   └── __init__.py      # App factory
├── configs/             # Configuration
│   ├── __init__.py
│   └── config.py
├── library/             # Core processing library
├── utils/               # Utilities
│   ├── __init__.py
│   └── device_manager.py
└── assets/              # Static assets
    ├── languages/
    ├── models/
    └── binary/
```

### Key Components

1. **DeviceManager** - Automatic device detection and management
2. **ConfigManager** - Configuration loading and validation
3. **App Factory** - Gradio application creation
4. **CLI** - Command-line interface with rich output

## Dependencies

### Core Dependencies
- torch >= 1.12.0
- torchaudio >= 0.12.0  
- numpy >= 1.21.0
- gradio >= 3.30.0
- librosa >= 0.9.0
- scipy >= 1.7.0
- onnxruntime >= 1.12.0

### Audio Processing
- soundfile >= 0.12.0
- pydub >= 0.25.0
- resampy >= 0.4.0

### Utilities
- Pillow >= 8.0.0
- click >= 8.0.0
- psutil >= 5.8.0
- tqdm >= 4.60.0

### Optional GPU Support
- torch-directml (Windows)
- pytorch-ocl (Linux)  
- onnxruntime-gpu

## Troubleshooting

### Device Detection Issues

```bash
# Force device detection
terauvr devices --verbose

# Check device health
terauvr devices --format json | jq
```

### Configuration Problems

```bash
# Reset to defaults
terauvr config generate --output config.yaml

# Validate configuration
terauvr config show --format json
```

### Performance Issues

1. **Memory**: Increase `gpu_memory_threshold` or reduce `batch_size`
2. **CPU**: Set `cpu_mode: true` to disable GPU usage
3. **DirectML**: Try switching to CPU mode if unstable

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black terauvr/

# Lint code  
flake8 terauvr/

# Type checking
mypy terauvr/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/terastudio-org/terauvr/issues)
- **Documentation**: [ReadTheDocs](https://terastudio-docs.readthedocs.io/)
- **Discussions**: [GitHub Discussions](https://github.com/terastudio-org/terauvr/discussions)

## Acknowledgments

- Original UVR team for the base models
- PyTorch and ONNX communities
- Gradio team for the web framework
- Contributors and testers

---

**TeraStudio UVR** - Making audio separation accessible to everyone.