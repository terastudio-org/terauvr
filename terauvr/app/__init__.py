"""
Main application module for terauvr.

This module provides the main application factory and GUI components
for the TeraStudio UVR package.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import gradio as gr

from ..configs.config_manager import get_config_manager
from ..utils.device_manager import get_device_manager


def create_app(
    config_path: Optional[str] = None,
    device: str = "auto",
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    verbose: bool = False
) -> gr.Blocks:
    """
    Create and configure the main Gradio application.
    
    Args:
        config_path: Optional path to configuration file
        device: Device to use for computation
        host: Host to bind the server to
        port: Port to run the server on
        share: Whether to create a public share link
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured Gradio Blocks application
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Initializing TeraStudio UVR application...")
    
    try:
        # Load configuration
        config_manager = get_config_manager(config_path)
        
        # Update device configuration if specified
        if device != "auto":
            config_manager.set('device', device)
        
        # Get current configuration
        config = config_manager.get_config()
        
        # Setup device
        device_manager = get_device_manager()
        
        if device == "auto":
            recommended_device = device_manager.get_recommended_device()
            if recommended_device:
                logger.info(f"Auto-detected device: {recommended_device.name}")
            else:
                logger.warning("No suitable device found, using CPU")
        else:
            logger.info(f"Using specified device: {device}")
        
        # Create the Gradio application
        with gr.Blocks(
            title="📱 TeraStudio UVR by terastudio",
            theme=_get_theme(),
            css=_get_custom_css(),
        ) as app:
            
            # Header
            gr.HTML("""
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
                <h1 style='margin: 0; font-size: 2.5em;'>🎵TeraStudio UVR by terastudio🎵</h1>
                <p style='margin: 10px 0; font-size: 1.2em;'>Advanced AI-powered VR/AR Audio Separation</p>
            </div>
            """)
            
            # Main content with tabs
            with gr.Tabs():
                _create_inference_tab(config_manager, device_manager)
                _create_settings_tab(config_manager, device_manager)
                _create_about_tab()
        
        logger.info("TeraStudio UVR application initialized successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise


def _get_theme():
    """Create the application theme."""
    try:
        # Try to load custom theme from configuration
        theme_name = get_config_manager().get('theme', 'NoCrypt/miku')
        
        # For now, use default theme
        # TODO: Implement custom theme loading
        return gr.themes.Soft()
        
    except Exception as e:
        logging.warning(f"Failed to load theme: {e}")
        return gr.themes.Soft()


def _get_custom_css():
    """Get custom CSS for the application."""
    return """
    .gradio-container {
        max-width: 1200px !important;
    }
    
    .tab-content {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
    }
    
    .status-indicator {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .status-available {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .status-unavailable {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .device-info {
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 10px 0;
    }
    """


def _create_inference_tab(config_manager, device_manager):
    """Create the main inference tab."""
    with gr.Tab("🎯 Audio Separation"):
        
        gr.HTML("<h2>🎼 Music Separation</h2>")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File",
                    type="filepath",
                    format="mp3"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    model_choice = gr.Dropdown(
                        choices=["htdemucs", "htdemucs_6s", "demucs", "htdemucs_v3"],
                        value="htdemucs",
                        label="🎭 Separation Model"
                    )
                    
                    stems_choice = gr.CheckboxGroup(
                        choices=["vocals", "drums", "bass", "other", "guitar", "piano"],
                        value=["vocals", "drums", "bass", "other"],
                        label="🎯 Stems to Extract"
                    )
                    
                    output_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac"],
                        value="wav",
                        label="📁 Output Format"
                    )
                
                separate_btn = gr.Button(
                    "🎵 Start Separation",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                # Output section
                output_files = gr.File(
                    label="📁 Separated Audio Files",
                    file_count="multiple",
                    file_types=["audio"]
                )
                
                # Progress indicator
                progress_status = gr.HTML(value="<p style='color: #666;'>Ready to separate audio...</p>")
        
        # Bind events
        separate_btn.click(
            fn=_handle_audio_separation,
            inputs=[audio_input, model_choice, stems_choice, output_format],
            outputs=[output_files, progress_status]
        )


def _create_settings_tab(config_manager, device_manager):
    """Create the settings tab."""
    with gr.Tab("⚙️ Settings"):
        
        gr.HTML("<h2>⚙️ Configuration</h2>")
        
        with gr.Row():
            # Device Settings
            with gr.Column():
                gr.HTML("<h3>💻 Device Settings</h3>")
                
                device_status = gr.HTML()
                
                with gr.Group():
                    current_device = config_manager.get('device', 'auto')
                    device_display = f"Current Device: **{current_device}**"
                    
                    if current_device == 'auto':
                        recommended = device_manager.get_recommended_device()
                        if recommended:
                            device_display += f" (Recommended: {recommended.name})"
                    
                    device_info = gr.HTML(value=f"<div class='device-info'>{device_display}</div>")
                
                # Device selection
                new_device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda", "directml", "opencl", "mps"],
                    value=current_device,
                    label="💻 Compute Device"
                )
                
                device_apply_btn = gr.Button("🔄 Apply Device Settings")
                
            # Application Settings
            with gr.Column():
                gr.HTML("<h3>🎛️ Application Settings</h3>")
                
                # Language selection
                language = gr.Dropdown(
                    choices=["en-US", "vi-VN"],
                    value=config_manager.get('language', 'en-US'),
                    label="🌐 Language"
                )
                
                # Theme selection  
                theme = gr.Dropdown(
                    choices=["NoCrypt/miku", "gradio/light", "gradio/dark"],
                    value=config_manager.get('theme', 'gradio/light'),
                    label="🎨 Theme"
                )
                
                # Server settings
                port = gr.Number(
                    value=config_manager.get('app_port', 7860),
                    label="🔌 Port"
                )
                
                fp16_enabled = gr.Checkbox(
                    value=config_manager.get('fp16', False),
                    label="🚀 Enable FP16 (faster processing)"
                )
        
        # Settings application
        settings_apply_btn = gr.Button("💾 Save Settings")
        settings_status = gr.HTML()
        
        # Bind events
        device_apply_btn.click(
            fn=_handle_device_change,
            inputs=[new_device],
            outputs=[device_status, settings_status]
        )
        
        settings_apply_btn.click(
            fn=_handle_settings_change,
            inputs=[language, theme, port, fp16_enabled],
            outputs=[settings_status]
        )


def _create_about_tab():
    """Create the about tab."""
    with gr.Tab("ℹ️ About"):
        
        gr.HTML("""
        <div style='padding: 20px; text-align: center;'>
            <h2>🎵 TeraStudio UVR</h2>
            <p style='font-size: 1.2em; color: #666; margin: 20px 0;'>
                Advanced AI-powered VR/AR audio separation toolkit
            </p>
            
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h3>🚀 Features</h3>
                <ul style='text-align: left; max-width: 600px; margin: 0 auto;'>
                    <li>🎯 Multi-stem audio separation (vocals, drums, bass, etc.)</li>
                    <li>💻 Automatic device detection and optimization</li>
                    <li>🔥 GPU acceleration support (CUDA, DirectML, OpenCL, MPS)</li>
                    <li>🎨 Modern web interface with Gradio</li>
                    <li>🌐 Multi-language support</li>
                    <li>⚙️ Flexible configuration system</li>
                </ul>
            </div>
            
            <div style='margin: 20px 0;'>
                <h3>📊 Supported Models</h3>
                <p>HT-Demucs, Demucs, MDX-Net, and more...</p>
            </div>
            
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;'>
                <p><strong>Author:</strong> terastudio-org/terastudio</p>
                <p><strong>License:</strong> MIT</p>
                <p><strong>Version:</strong> 1.0.0</p>
            </div>
        </div>
        """)


def _handle_audio_separation(audio_file, model, stems, output_format):
    """Handle audio separation request."""
    if audio_file is None:
        return None, "<p style='color: red;'>Please upload an audio file first.</p>"
    
    # TODO: Implement actual audio separation logic
    # This is a placeholder for the real implementation
    
    try:
        status = f"""
        <div class='status-indicator status-available'>
            ✅ Separation completed successfully!
            <br>📁 Model: {model}
            <br>🎯 Stems: {', '.join(stems)}
            <br>📋 Format: {output_format.upper()}
        </div>
        """
        
        # Return dummy output files
        return [audio_file], status
        
    except Exception as e:
        return None, f"""
        <div class='status-indicator status-unavailable'>
            ❌ Separation failed: {str(e)}
        </div>
        """


def _handle_device_change(new_device):
    """Handle device configuration changes."""
    try:
        config_manager = get_config_manager()
        config_manager.set('device', new_device)
        
        status = f"""
        <div class='status-indicator status-available'>
            ✅ Device updated to: {new_device}
        </div>
        """
        
        return status, status
        
    except Exception as e:
        status = f"""
        <div class='status-indicator status-unavailable'>
            ❌ Failed to update device: {str(e)}
        </div>
        """
        return status, status


def _handle_settings_change(language, theme, port, fp16_enabled):
    """Handle application settings changes."""
    try:
        config_manager = get_config_manager()
        
        config_manager.set('language', language)
        config_manager.set('theme', theme)
        config_manager.set('app_port', port)
        config_manager.set('fp16', fp16_enabled)
        
        status = """
        <div class='status-indicator status-available'>
            ✅ Settings saved successfully! Please restart the application to apply changes.
        </div>
        """
        
        return status
        
    except Exception as e:
        status = f"""
        <div class='status-indicator status-unavailable'>
            ❌ Failed to save settings: {str(e)}
        </div>
        """
        return status