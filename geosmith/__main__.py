"""Command-line entry point for GeoSmith."""

import sys


def main():
    """Main entry point."""
    print("GeoSmith: *Smith for ML models")
    print("Use geosmith as a Python library:")
    print("  import geosmith")
    print("  from geosmith import make_features, process_raster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
