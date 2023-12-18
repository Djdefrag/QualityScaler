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
. $(dirname $(abs_path ${BASH_SOURCE[0]}))/venv/bin/activate
export PATH=$(dirname $(abs_path ${BASH_SOURCE[0]}))/:$PATH

export IN_ACTIVATED_ENV="1"
