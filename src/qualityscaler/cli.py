"""
Main entry point.
"""

import os
import sys



def main() -> int:
    """Main entry point for the template_python_cmd package."""
    python_exe = sys.executable
    print(f"Python executable: {python_exe}")
    rtn = os.system(f"{python_exe} -m qualityscaler.QualityScaler")
    return rtn


if __name__ == "__main__":
    sys.exit(main())
