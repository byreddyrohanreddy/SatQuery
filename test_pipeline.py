#!/usr/bin/env python3
"""
CLI test script for SatQuery AI.

Runs the full pipeline end-to-end and pretty-prints the JSON response.
This is how we verify the system works before any UI exists.

Usage:
    # Single image
    python test_pipeline.py --images path/to/image.jpg --query "What is in this image?"

    # Two images (bi-temporal or optical+SAR)
    python test_pipeline.py --images img1.jpg img2.tif --query "What changed?"

    # Override modalities (useful for testing fusion with two optical images)
    python test_pipeline.py --images a.jpg b.jpg --query "Fuse these" --modalities optical SAR
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="SatQuery AI — CLI pipeline test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Path(s) to 1 or 2 image files",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language query",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help='Override auto-detected modalities (e.g. "optical" "SAR")',
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Validate image paths
    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"ERROR: Image file not found: {img_path}", file=sys.stderr)
            sys.exit(1)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Load images
    print(f"\n{'='*60}")
    print(f"  SatQuery AI - Pipeline Test")
    print(f"{'='*60}")
    print(f"  Query:      {args.query}")
    print(f"  Images:     {', '.join(args.images)}")
    if args.modalities:
        print(f"  Modalities: {', '.join(args.modalities)} (override)")
    print(f"{'='*60}\n")

    from satquery.validator import load_image_as_pil

    images = []
    filenames = []
    for img_path in args.images:
        try:
            img, fmt = load_image_as_pil(img_path)
            images.append(img)
            filenames.append(Path(img_path).name)
            print(f"  [+] Loaded: {img_path} ({img.size[0]}x{img.size[1]}, {img.mode}, {fmt})")
        except Exception as e:
            print(f"  [-] Failed to load {img_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print()

    # Import pipeline (after images are loaded to show load errors separately)
    from satquery.api import run_pipeline
    from satquery.models import ClarificationResponse, QueryResponse

    # Run pipeline
    print("  Running pipeline...")
    result = run_pipeline(
        images=images,
        query=args.query,
        filenames=filenames,
        modalities_override=args.modalities,
    )

    # Pretty-print result
    print(f"\n{'-'*60}")

    if isinstance(result, ClarificationResponse):
        print("  [!] CLARIFICATION NEEDED")
        print(f"{'-'*60}")
        output = result.model_dump()
    elif isinstance(result, QueryResponse):
        print("  [+] ANALYSIS COMPLETE")
        print(f"{'-'*60}")
        output = result.model_dump()
    else:
        output = result

    # JSON output with nice formatting
    print(json.dumps(output, indent=2, default=str))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
