.. _installation:

Installation Guide
===========================


Pre-Installation
----------------------------
Before you can install the software, you must first have a working Python distribution with a package manager. For all platforms we recommend that you download and install Anaconda - a professional grade, full-blown scientific Python distribution.

To set up Anaconda, you should:

    * Download and install Anaconda (Python 3.x version, 64 bit installer is recommended) from https://www.continuum.io/downloads
    
    * Update the root Anaconda environment (type in a terminal): 
        
        ``>> conda update --all``
    
    * Activate the Anaconda root environment in a terminal as follows: 
        
        ``>> activate``

It is recommended to create a new environment to install hydesign if you have other Python programs. This ensures that the dependencies for the different packages do not conflict with one another. In the command prompt, create and active an environment with::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign.git
   cd hydesign
   conda env create --file environment.yml
   conda activate hydesign

It is also recommended that you install cplex first before installing HyDesign. To get the community-edition of CPLEX you can run:

    ``pip install cplex``

If your combination of platform and Python version is not available on PyPi you need to go through the IBM website: https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing to download and install the package. Here you can also find the full license (not free) of CPLEX that can solve larger problems.


Simple Installation
----------------------------

hydesign’s base code is open-sourced and freely available on `GitLab 
<https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign>`_ (MIT license).

* Install from PyPi.org (official releases)::
  
    pip install hydesign

* Install from GitLab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign.git
        


Developer Installation
-------------------------------

We highly recommend developers to install hydesign into the environment created previously. The commands to clone and install hydesign with developer options into the current active environment in an Anaconda Prompt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign.git
   cd hydesign
   pip install -e .
