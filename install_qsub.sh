#!/bin/bash

# multi-core compilation
export COMPILE_JOBS=8

# environment variables
PAYNT_ROOT=`pwd`
PREREQUISITES=${PAYNT_ROOT}/prerequisites # modify this to install prerequisites outside of Paynt

# Load necessary modules (adjust based on available modules in Metacentrum)
module add cmake/3.23.1-gcc-10.2.1-gxvea6z
module add boost/1.79.0-gcc-10.2.1-am7oanb
# module add git
# module add python/3.10.4-gcc-8.3.0-ovkjwzd  # Přidání Pythonu
# Využíváme nainstalovanou python3.11
python3.11 -m ensurepip --upgrade --user
module add maven
module add eigen

# prerequisites
mkdir -p ${PREREQUISITES}

# Build cvc5 (optional)
# cd ${PREREQUISITES}
# git clone --depth 1 --branch cvc5-1.0.0 https://github.com/cvc5/cvc5.git cvc5
# cd ${PREREQUISITES}/cvc5
# ./configure.sh --prefix="." --auto-download --python-bindings
# cd build
# make --jobs ${COMPILE_JOBS}
# make install

# build storm
cd ${PREREQUISITES}
git clone https://github.com/moves-rwth/storm.git storm
mkdir -p ${PREREQUISITES}/storm/build
cd ${PREREQUISITES}/storm/build
cmake ..
make storm storm-pomdp storm-counterexamples --jobs ${COMPILE_JOBS}m

# setup and activate python environment
python3.11 -m venv ${PREREQUISITES}/venv
source ${PREREQUISITES}/venv/bin/activate
python3.11 -m pip3 install wheel

# build pycarl
cd ${PREREQUISITES}
git clone https://github.com/moves-rwth/pycarl.git pycarl
cd ${PREREQUISITES}/pycarl
python3.11 setup.py develop

# build stormpy
cd ${PREREQUISITES}
git clone https://github.com/moves-rwth/stormpy.git stormpy
cd ${PREREQUISITES}/stormpy
python3 setup.py develop

cd ${PREREQUISITES}
git clone https://github.com/kurecka/VecStorm VecStorm
cd ${PREREQUISITES}/VecStorm
python3 setup.py develop

# paynt dependencies
module add graphviz
python3.11 -m pip install click z3-solver psutil graphviz

# build payntbind
cd ${PAYNT_ROOT}/payntbind
python3 setup.py develop
cd ${PAYNT_ROOT}

python3.11 -m pip install tensorflow==2.15 tf_agents tqdm dill matplotlib pandas seaborn

# done
deactivate