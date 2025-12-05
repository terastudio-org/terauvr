#!/usr/bin/env python3
"""
Main entry point for the terauvr package.

This module provides the command-line interface and main application launcher.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from terauvr import __version__, __author__, get_info
from terauvr.cli import main as cli_main
from terauvr.app import create_app


def main():
    """Main entry point for the terauvr package."""
    parser = argparse.ArgumentParser(
        description="TeraStudio UVR - Advanced AI-powered VR/AR audio separation toolkit",
        epilog=f"Author: {__author__}\nVersion: {__version__}"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"terauvr {__version__}"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show package information"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true", 
        help="Launch the graphical user interface"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps", "directml", "opencl"],
        default="auto",
        help="Specify device for computation (default: auto-detect)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Suppress output except for errors"
    )
    
    args = parser.parse_args()
    
    # Handle info flag
    if args.info:
        info = get_info()
        print("TeraStudio UVR Package Information:")
        print(f"  Name: {info['name']}")
        print(f"  Version: {info['version']}") 
        print(f"  Author: {info['author']}")
        print(f"  License: {info['license']}")
        print(f"  Homepage: {info['homepage']}")
        return
    
    # Setup logging
    log_level = logging.WARNING
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        if args.gui:
            # Launch GUI application
            logger = logging.getLogger(__name__)
            logger.info("Launching TeraStudio UVR GUI...")
            
            app = create_app(
                config_path=args.config,
                device=args.device,
                verbose=args.verbose > 0
            )
            
            app.launch()
        else:
            # Run CLI mode
            cli_main(args)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        if args.verbose > 0:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()