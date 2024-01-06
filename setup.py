"""
Setup file.
"""

import os
import re

from setuptools import setup

URL = "https://github.com/zackees/QualityScaler"
KEYWORDS = "upscales images using ai"
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        keywords=KEYWORDS,
        long_description_content_type="text/markdown",
        url=URL,
        include_package_data=False)
