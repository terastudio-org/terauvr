"""
Command Line Interface for terauvr package.

This module provides a comprehensive CLI for the TeraStudio UVR package
with device detection, configuration management, and audio processing capabilities.
"""

import sys
import argparse
import logging
from typing import Optional
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from terauvr.utils.device_manager import detect_devices, get_recommended_device, get_best_device
from terauvr.configs.config_manager import get_config_manager


console = Console()


@click.group()

@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--quiet", "-q",
    is_flag=True, 
    help="Suppress all output except errors"
)
@click.pass_context
def cli(ctx, verbose, quiet):
    """TeraStudio UVR - Advanced AI-powered VR/AR audio separation toolkit."""
    # Setup logging
    if quiet:
        logging.basicConfig(level=logging.ERROR)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.option(
    "--prefer-gpu/--no-gpu",
    default=True,
    help="Whether to prefer GPU acceleration"
)
@click.option(
    "--format",
    type=click.Choice(['table', 'json']),
    default='table',
    help="Output format"
)
def devices(prefer_gpu, format):
    """Detect and display available compute devices."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=console.is_terminal is False
        ) as progress:
            task = progress.add_task("Detecting devices...", total=None)
            
            devices = detect_devices()
            recommended = get_recommended_device(prefer_gpu)
            best = get_best_device()
        
        if format == 'json':
            import json
            device_data = []
            for device in devices:
                device_data.append({
                    'type': device.device_type.value,
                    'id': device.device_id,
                    'name': device.name,
                    'memory_gb': device.memory_gb,
                    'compute_capability': device.compute_capability,
                    'available': device.is_available,
                    'recommended': recommended and device.device_id == recommended.device_id and device.device_type == recommended.device_type,
                    'best': best and device.device_id == best.device_id and device.device_type == best.device_type
                })
            
            console.print(json.dumps(device_data, indent=2))
            
        else:
            # Table format
            table = Table(title="TeraStudio UVR - Device Detection")
            table.add_column("Type", style="cyan", no_wrap=True)
            table.add_column("ID", style="magenta")
            table.add_column("Name", style="green")
            table.add_column("Memory", style="yellow")
            table.add_column("Compute", style="blue")
            table.add_column("Status", style="red" if "Not Available" else "green")
            table.add_column("Flags", style="cyan")
            
            for device in devices:
                memory_str = f"{device.memory_gb:.1f}GB" if device.memory_gb else "N/A"
                compute_str = device.compute_capability or "N/A"
                status_str = "Available" if device.is_available else "Not Available"
                
                flags = []
                if recommended and device.device_id == recommended.device_id and device.device_type == recommended.device_type:
                    flags.append("RECOMMENDED")
                if best and device.device_id == best.device_id and device.device_type == best.device_type:
                    flags.append("BEST")
                
                flags_str = ", ".join(flags) if flags else ""
                
                table.add_row(
                    device.device_type.value.upper(),
                    str(device.device_id),
                    device.name,
                    memory_str,
                    compute_str,
                    status_str,
                    flags_str
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Error detecting devices: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output configuration file path"
)
@click.option(
    "--format",
    type=click.Choice(['yaml', 'json']),
    default='yaml',
    help="Configuration file format"
)
@click.option(
    "--device",
    type=click.Choice(['auto', 'cpu', 'cuda', 'directml', 'opencl', 'mps']),
    default='auto',
    help="Target compute device"
)
@click.option(
    "--port",
    type=int,
    default=7860,
    help="Application port"
)
@click.option(
    "--language",
    default='en-US',
    help="UI language"
)
def config(output, format, device, port, language):
    """Generate or modify configuration."""
    try:
        config_manager = get_config_manager()
        
        # Update configuration
        if device != 'auto':
            config_manager.set('device', device)
        if port != 7860:
            config_manager.set('app_port', port)
        if language != 'en-US':
            config_manager.set('language', language)
        
        # Save configuration
        output_path = output or f"config.{format}"
        config_manager.save(output_path)
        
        console.print(f"[green]Configuration saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error configuring: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--format",
    type=click.Choice(['yaml', 'json']),
    default='yaml',
    help="Configuration file format"
)
def show_config(format):
    """Display current configuration."""
    try:
        config_manager = get_config_manager()
        config_dict = config_manager.to_dict()
        
        if format == 'json':
            import json
            console.print_json(data=config_dict)
        else:
            # YAML format
            import yaml
            yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2)
            console.print(Panel(yaml_content, title="Current Configuration"))
            
    except Exception as e:
        console.print(f"[red]Error showing configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument(
    'input_file',
    type=click.Path(exists=True)
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(),
    default='./output',
    help='Output directory for separated tracks'
)
@click.option(
    '--model', '-m',
    type=str,
    default='htdemucs',
    help='Model to use for separation'
)
@click.option(
    '--device', '-d',
    type=click.Choice(['auto', 'cpu', 'cuda', 'directml', 'opencl', 'mps']),
    default='auto',
    help='Device to use for processing'
)
@click.option(
    '--stems',
    type=str,
    default='drums,bass,other,vocals',
    help='Comma-separated list of stems to extract'
)
def separate(input_file, output_dir, model, device, stems):
    """Separate audio into individual tracks."""
    try:
        input_path = Path(input_file)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            console.print(f"[red]Input file not found: {input_file}[/red]")
            sys.exit(1)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Separating audio...", total=None)
            
            # TODO: Implement actual separation logic here
            # This is a placeholder for the actual implementation
            console.print(f"[yellow]Separation not yet implemented[/yellow]")
            console.print(f"[yellow]Input: {input_file}[/yellow]")
            console.print(f"[yellow]Output: {output_dir}[/yellow]")
            console.print(f"[yellow]Model: {model}[/yellow]")
            console.print(f"[yellow]Device: {device}[/yellow]")
            console.print(f"[yellow]Stems: {stems}[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error separating audio: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--port",
    type=int,
    default=7860,
    help="Port to run the server on"
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to"
)
@click.option(
    "--share",
    is_flag=True,
    help="Create a public share link"
)
@click.option(
    "--device",
    type=click.Choice(['auto', 'cpu', 'cuda', 'directml', 'opencl', 'mps']),
    default='auto',
    help="Device to use for processing"
)
@click.option(
    "--config",
    type=click.Path(),
    help="Configuration file path"
)
def gui(port, host, share, device, config):
    """Launch the graphical user interface."""
    try:
        from ..app import create_app
        
        console.print("[green]Starting TeraStudio UVR GUI...[/green]")
        
        app = create_app(
            config_path=config,
            device=device,
            host=host,
            port=port,
            share=share
        )
        
        console.print(f"[green]Server running on http://{host}:{port}[/green]")
        if share:
            console.print("[green]Public share link will be available[/green]")
        
        app.launch()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting GUI: {e}[/red]")
        sys.exit(1)


@cli.command()
def info():
    """Display package information."""
    console.print(Panel.fit(
        f"[bold green]TeraStudio UVR[/bold green]\n"
        f"[bold blue]Version:[/bold blue] {__version__}\n"
        f"[bold blue]Author:[/bold blue] terastudio-org/terastudio\n"
        f"[bold blue]License:[/bold blue] MIT\n\n"
        f"[bold yellow]Advanced AI-powered VR/AR audio separation toolkit[/bold yellow]",
        title="Package Information"
    ))


# Legacy main function for backward compatibility
def main(args: Optional[list] = None):
    """Main entry point for CLI."""
    if args is None:
        args = sys.argv[1:]
    
    try:
        cli.main(standalone_mode=False, args=args)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]CLI error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
