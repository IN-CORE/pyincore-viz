#!/bin/bash

TESTPYPI="--extra-index-url https://test.pypi.org/simple/"
#TESTPYPI="--index-url http://localhost:3141/incore/dev"

# setup pip-compile to resolve dependencies, also is minimal version of python
if [ ! -d /tmp/pyincore-requirements ]; then
  python3 -m virtualenv /tmp/pyincore-requirements
  . /tmp/pyincore-requirements/bin/activate
  pip install pip-tools
else
  . /tmp/pyincore-requirements/bin/activate
fi

# all requirements in pyincore_viz files
IMPORTS=$(egrep -R -h --include "*.py" '(import|from) ' pyincore_viz | \
          sed -e 's/^ *"//' -e 's/\\n",$//' | \
          egrep '^(from|import)' | \
          awk '!/pyincore-viz/ { print $2 }' | \
          sort -u | \
          egrep -v '^(\(|_)' | \
          sed 's/,/ /g')
# check which imports are not standard python
rm -f requirements.tmp
for x in $IMPORTS; do
  python3 -c "import $x" 2>&1 | \
    awk '/ModuleNotFoundError/ { print $5 }' | \
    sed -e 's/yaml/pyyaml/' -e 's/jose/python-jose/' -e 's/_pytest/pytest/' -e 's/PIL/pillow/' -e 's/osgeo/gdal/' -e "s/'//g" >> requirements.tmp
done
sort -u requirements.tmp > requirements.pyincore_viz

# all requirements in test files
IMPORTS=$(egrep -R -h --include "*.py" '(import|from) ' tests | \
          sed -e 's/^ *"//' -e 's/\\n",$//' | \
          egrep '^(from|import)' | \
          awk '!/pyincore-viz/ { print $2 }' | \
          sort -u | \
          egrep -v '^(\(|_)' | \
          sed 's/,/ /g')
# check which imports are not standard python
rm -f requirements.tmp
for x in $IMPORTS; do
    MISSING=$(python3 -c "import $x" 2>&1 | \
      awk '/ModuleNotFoundError/ { print $5 }' | \
      sed -e 's/yaml/pyyaml/' -e 's/jose/python-jose/' -e 's/_pytest/pytest/' -e 's/PIL/pillow/' -e 's/osgeo/gdal/' -e "s/'//g")
    if ! grep "${MISSING}" requirements.pyincore_viz &>/dev/null ; then
      echo ${MISSING} >> requirements.tmp
    fi
done
sort -u requirements.tmp > requirements.testing

# combine pyincore_viz and testing
cat requirements.pyincore_viz requirements.testing > requirements.in

# create the requirements.txt file for pip. This is intended to setup a virtualenv for
# development on pyincore_viz.
pip-compile ${TESTPYPI} --quiet --upgrade --rebuild --output-file requirements.txt requirements.in
cat requirements.txt | grep -v ' *#.*' | grep -v '^$' | grep -v "^${TESTPYPI}$" > requirements.ver

# create the environment.yml file for conda. This is intended to setup a conda environment for
# development on pyincore_viz.
cat << EOF > environment.yml
name: base
channels:
  - conda-forge
  - defaults
dependencies:
  - ipopt>=3.11
  - numpy>=1.16
EOF
cat requirements.ver | egrep -v '^(numpy|ipopt)==' | sed 's/^/  - /' >> environment.yml

# update conda recipe
for x in $(cat requirements.pyincore_viz | egrep -v '(ipopt|numpy)'); do
  if ! grep "    - $x==" recipes/meta.yaml >/dev/null ; then
    echo "CONDA NEW IMPORT $x"
  fi
  version=$(grep "^$x==" requirements.ver)
  sed -i~ "s/    - $x==.*/    - $version/" recipes/meta.yaml
done
for x in $(cat requirements.testing | egrep -v '(pytest)'); do
  if ! grep "    - $x==" recipes/meta.yaml >/dev/null ; then
    echo "CONDA TEST NEW IMPORT $x"
  fi
  version=$(grep "^$x==" requirements.ver)
  sed -i~ "s/    - $x==.*/    - $version/" recipes/meta.yaml
done

# update setup file
for x in $(cat requirements.pyincore_viz | egrep -v '(ipopt)'); do
  if ! grep "'$x==" setup.py >/dev/null ; then
    echo "SETUP NEW IMPORT $x"
  fi
  version=$(grep "^$x==" requirements.ver)
  sed -i~ "s/'$x==.*'/'$version'/" setup.py
done
for x in $(cat requirements.testing); do
  if ! grep "'$x==" setup.py >/dev/null ; then
    echo "SETUP TEST NEW IMPORT $x"
  fi
  version=$(grep "^$x==" requirements.ver)
  sed -i~ "s/'$x==.*'/'$version'/" setup.py
done

# cleanup
rm -f requirements.pyincore_viz requirements.testing requirements.notebooks requirements.tmp requirements.ver
