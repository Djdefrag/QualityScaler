"""

  curl https://raw.githubusercontent.com/zackees/install.py/main/install.py | python
  To enter the environment run:
    source activate.sh


  Notes:
    This script is tested to work using python2 and python3 from a fresh install. The only side effect
    of running this script is that virtualenv will be globally installed if it isn't already.
"""


import argparse
import os
import re
import shutil
import subprocess
import sys
import warnings
from shutil import which as find_executable
from typing import Optional

IS_GITHUB = 'GITHUB_WORKSPACE' in os.environ

# This activation script adds the ability to run it from any path and also
# aliasing pip3 and python3 to pip/python so that this works across devices.
_ACTIVATE_SH = """
#!/bin/bash
set -e

# Get the directory of the current script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to that directory
cd "$script_dir"

# Confirm the current directory
echo "Changed directory to: $(pwd)"
echo "activate.sh change working directory to $(pwd)"


if [ "$IN_ACTIVATED_ENV" = "1" ]; then
  IN_ACTIVATED_ENV=1
else
  IN_ACTIVATED_ENV=0
fi

# If the 'venv' directory doesn't exist, print a message and exit.
if [ ! -d "venv" ]; then
  cwd=$(pwd)
  echo "The 'venv' directory in $cwd does not exist, creating..."
  echo "OSTYPE: $OSTYPE"
  case "$OSTYPE" in
    darwin*|linux-gnu*)
      python3 ./install.py
      ;;
    *)
      python ./install.py
      ;;
  esac

  . ./venv/bin/activate
  export IN_ACTIVATED_ENV=1
  this_dir=$(pwd)
  export PATH="$this_dir:$PATH"
  echo "Environment created."
  pip install -e .
  exit 0
fi

if [[ "$IN_ACTIVATED_ENV" != "1" ]]; then
  . ./venv/bin/activate
  export IN_ACTIVATED_ENV=1
  this_dir=$(pwd)
  export PATH="$this_dir:$PATH"
fi
"""
HERE = os.path.dirname(__file__)
os.chdir(os.path.abspath(HERE))

print(f"install.py changed directory to {os.getcwd()}")


def _exe(cmd: str, check: bool = True, cwd: Optional[str] = None) -> None:
    msg = (
        "########################################\n"
        f"# Executing '{cmd}'\n"
        "########################################\n"
    )
    print(msg)
    sys.stdout.flush()
    sys.stderr.flush()
    # os.system(cmd)
    subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def is_tool(name):
    """Check whether `name` is on PATH."""

    return find_executable(name) is not None


def platform_ensure_python_installed() -> None:
    try:
        python_x = "python" if sys.platform == "win32" else "python3"
        stdout = subprocess.check_output(
            [python_x, "--version"], universal_newlines=True
        )
        print(f"Python is already installed: {stdout}")
        return
    except Exception:
        pass
    if sys.platform == "darwin":
        _exe("brew install python3")
    elif sys.platform == "linux":
        _exe("sudo apt-get install python3")
    elif sys.platform == "win32":
        _exe("choco install python3")


def get_pip() -> str:
    if sys.platform == "win32":
        return "pip"
    return "pip3"


def get_python() -> str:
    return sys.executable


def create_virtual_environment() -> None:
    try:
        _exe(f"{get_python()} -m venv venv")
    except subprocess.CalledProcessError as exc:
        warnings.warn(f"couldn't make virtual environment because of {exc}")
        raise exc


def check_platform() -> None:
    if sys.platform == "win32":
        is_git_bash = os.environ.get("ComSpec", "").endswith("bash.exe")
        if not is_git_bash:
            print("This script only works with git bash on windows.")
            sys.exit(1)

def convert_windows_path_to_git_bash_path(path: str) -> str:
    # Function to replace the matched drive letter and colon
    def replace_drive_letter(match):
        return "/" + match.group(1).lower() + "/"

    # Replace the drive letter and colon with Git Bash format
    path = re.sub(r"([A-Za-z]):\\", replace_drive_letter, path)
    # Replace backslashes with forward slashes
    path = path.replace("\\", "/")
    return path


def modify_activate_script() -> None:
    path = os.path.join(HERE, "venv", "bin", "activate")
    abs_path = os.path.abspath(path)
    if sys.platform == "win32":
        abs_path = convert_windows_path_to_git_bash_path(abs_path)
    text_to_add = f'\nPATH="{abs_path}:$PATH"\n' + "export PATH"
    with open(path, encoding="utf-8", mode="a") as fd:
        fd.write(text_to_add)


def main() -> int:
    in_activated_env = os.environ.get("IN_ACTIVATED_ENV", "0") == "1"
    if in_activated_env:
        print(
            "Cannot install a new environment while in an activated environment. Please launch a new shell and try again."
        )
        return 1
    platform_ensure_python_installed()
    parser = argparse.ArgumentParser(description="Install the project.")
    parser.add_argument(
        "--remove", action="store_true", help="Remove the virtual environment"
    )
    args = parser.parse_args()
    if args.remove:
        print("Removing virtual environment")
        shutil.rmtree("venv", ignore_errors=True)
        return 0
    if not os.path.exists("venv"):
        create_virtual_environment()
    else:
        print(f'{os.path.abspath("venv")} already exists')
    # Is this necessary?
    os.chdir(os.path.abspath(HERE))
    activate_sh = os.path.join(HERE, "activate.sh")
    if not os.path.exists(activate_sh):
      with open(activate_sh, encoding="utf-8", mode="w") as fd:
          fd.write(_ACTIVATE_SH)
      if sys.platform != "win32":
          _exe(f'chmod +x {activate_sh}')
      _exe(f'git add {activate_sh} --chmod=+x')



    # Linux/MacOS uses bin and Windows uses Script, so create
    # a soft link in order to always refer to bin for all
    # platforms.
    if sys.platform == "win32" and not os.path.exists(os.path.join(HERE, "venv", "bin")):
        print("Creating soft link for windows")
        target = os.path.join(HERE, "venv", "Scripts")
        link = os.path.join(HERE, "venv", "bin")
        if not os.path.exists(link):
            _exe(f'mklink /J "{link}" "{target}"', check=False)

    
    assert os.path.exists(activate_sh), f"{activate_sh} does not exist"
    modify_activate_script()
    # Note that we can now just use pip instead of pip3 because
    # we are now in the virtual environment.
    try:
        cmd = "bash activate.sh" if sys.platform == "win32" else f'"{activate_sh}"'
        _exe(f"{cmd} && pip install -e .")  # Why does this fail on windows git-bash?
        print(
            'Now use ". ./activate.sh" (at the project root dir) to enter into the environment.'
        )
        return 0
    except subprocess.CalledProcessError:
        if IS_GITHUB:
            raise
        print(
            "Now complete install with `. ./activate.sh && pip install -e .`\n"
            "then use `. ./activate.sh` to enter into the environment."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
