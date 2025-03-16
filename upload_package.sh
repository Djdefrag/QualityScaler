#!/bin/bash
set -e
rm -rf build dist
uv pip install wheel twine
uv build --wheel
uv run twine upload dist/* --verbose
# echo Pushing git tags…
