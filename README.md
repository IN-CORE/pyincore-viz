# pyincore-viz

**pyincore-viz** is a Python project that provides visualization and other utilities for use with **pyincore**
The development is part of NIST sponsored IN-CORE (Interdependent Networked Community Resilience Modeling
Environment) initiative. 

**Mac specific notes**
    
- We use `matplotlib` library to create graphs. There is a Mac specific installation issue addressed at [here](https://stackoverflow.com/questions/4130355/python-matplotlib-framework-under-macosx) and 
[here](https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python). In a nutshell, 
insert line: `backend : Agg` into `~/.matplotlib/matplotlibrc` file.

