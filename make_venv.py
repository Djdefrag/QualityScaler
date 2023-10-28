"""
  Quick install
  cd <YOUR DIRECTORY>
  Download and install in one line:
    curl -X GET https://raw.githubusercontent.com/zackees/make_venv/main/make_venv.py | python

  To enter the environment run:
    source activate.sh

  Notes:
    This script is tested to work using python2 and python3 from a fresh install. The only side
    effect of running this script is that virtualenv will be globally installed if it isn't
    already.
"""

# pylint: disable=all

import os
import shutil
import subprocess
import sys

# This activation script adds the ability to run it from any path and also
# aliasing pip3 and python3 to pip/python so that this works across devices.
_ACTIVATE_SH = """
#!/bin/bash
#!/bin/bash
set -e
function abs_path {
  (cd "$(dirname '$1')" &>/dev/null && printf "%s/%s" "$PWD" "${1##*/}")
}

if [[ $(uname -a) == *"Microsoft"* ]]; then
  echo "Running on Windows"
else
  echo "Running on $(uname -a)"
  alias python=python3
  alias pip=pip3
fi

# if make_venv dir is not present, then make it
if [ ! -d "venv" ]; then
  python make_venv.py
fi
. $( dirname $(abs_path ${BASH_SOURCE[0]}))/venv/bin/activate
export PATH=$( dirname $(abs_path ${BASH_SOURCE[0]}))/:$PATH

export IN_ACTIVATED_ENV="1"
"""

HERE = os.path.dirname(__file__)
os.chdir(os.path.abspath(HERE))


def _exe(cmd):
    print('Executing "%s"' % cmd)
    # os.system(cmd)
    subprocess.check_call(cmd, shell=True)


def is_tool(name):
    """Check whether `name` is on PATH."""
    from distutils.spawn import find_executable

    return find_executable(name) is not None


shutil.rmtree("venv", ignore_errors=True)

if not is_tool("virtualenv"):
    _exe("pip install virtualenv")
# Which one is better? virtualenv or venv? This may switch later.
_exe("virtualenv -p python3 venv")
# _exe('python3 -m venv venv')
# Linux/MacOS uses bin and Windows uses Script, so create
# a soft link in order to always refer to bin for all
# platforms.
if sys.platform == "win32":
    target = os.path.join(HERE, "venv", "Scripts")
    link = os.path.join(HERE, "venv", "bin")
    _exe('mklink /J "%s" "%s"' % (link, target))
with open("activate.sh", "wt") as fd:
    fd.write(_ACTIVATE_SH)


print(
    'Now use ". activate.sh" (at the project root dir) to enter into the environment.'
)
