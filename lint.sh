#!/bin/bash
set -e
# cd to self bash script directory
cd $( dirname ${BASH_SOURCE[0]})
. ./activate.sh
echo Running ruff src
ruff --fix src
echo Running ruff tests
ruff --fix tests
echo Running black src tests
black src tests
echo Running isort src tests
isort --profile black src tests
echo Running flake8 src tests
flake8 src tests
echo Running pylint src
pylint src tests
echo Running mypy src
mypy src tests
echo Linting complete!
