## pyIncore_viz documentation

**pyIncore_viz** documentation which is built using Python [Sphinx](http://www.sphinx-doc.org/en/master/) package.

### Installation

Clone the code from pyincore repository [git](https://opensource.ncsa.illinois.edu/bitbucket/scm/incore1/pyincore_viz.git) 
repository.

### Building and running Sphinx in Docker container

Install [Docker Desktop](https://www.docker.com/) for your OS and change directory to your local branch `pyincore_viz` folder (one with Dockerfile).

1. Build container
    ```
    docker build -f Dockerfile.docs --no-cache -t pyincore_viz_docs .
    ```
    The container's name is **pyincore_viz_docs** in this example.
    
2. Run docker
    ```
    docker run --rm -p 80:80 --name doctest pyincore_viz_docs:latest
    ```
    Optional flag, `--name` sets container's name to **doctest** under which it appears in Docker Desktop.
   
3. Run html pages in your local browser (you might see the nginx main page first)
    ```
    http://localhost/doc/pyincore_viz/
    ``` 


### Running Sphinx directly in your environment

1. Install required packages. Currently `sphinx`, a Python package for building documentation and `sphinx_rtd_theme`, 
a theme used in this documentation and other packages. See section 4. for the full list.

2. We recommend using virtual environments, `conda` (preferred) or `virtualenv` for Python 3.7+. 
for managing Python environments.  
In case of `conda`, the package management and deployment tool 
is called `anaconda`. Create the environment from the terminal at the project 
folder (called `pyincore_viz` here) and activate it:
    ```
    conda create -n pyincore_viz python=3.7
    conda activate pyincore_viz
    ```
    or  
    ```
    virtualenv --python=python3.7 pyincore_viz
    source venv/bin/activate
    ```
   
3. Install required packages individually if necessary. Use `conda` again or you can also use `pip`:

    ```
    conda install sphinx
    conda install sphinx_rtd_theme
    conda install -c conda-forge nbsphinx
    ```
    or (global install for all users drop the --user flag)
    ```
    python3 -m pip install sphinx --user
    python3 -m pip install sphinx_rtd_theme --user
    python3 -m pip install nbsphinx --user
    ```   

4. From the terminal at the project folder (**pyincore/docs**) run: 
    ```
    sphinx-build -b html source build
    ```
    after that you should be able to run (`clean` deletes content of the `build` folder) :
    ```
    make clean
    make html
    ```
   
5. Open `index.html` from `build/html` directory in a browser.