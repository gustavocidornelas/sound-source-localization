# lmlib

lmlib is a efficient signal processing library for python. 
It modifies and transforms signal model approximations into a feature spaces.

## Installation
The ``lmlib`` package can be installed in different ways. 

### Installation for user
**------------TODO-------------** 

#### via ``pip``
```bash
$ pip install lmlib
```

#### Manually

1. download and unpack ``lmlib-master.zip``
2. go to ``lmlib-master``
3. Run ``python setup.py``


### Installation for developers
using gitlab

```bash
$ git clone https://gitlab.ti.bfh.ch/waldf1/lmlib
$ cd lmlib
$ pip install -r requirements.txt
$ pip install -e .
```
## Documentation
### Generate documentation
To generate the *html* documentation. Proceed as follow:

Open the terminal/bash in the root directory of `lmlib`
1. go to `doc` dir (`cd doc`)
2. type ``make html``
3. go to ``cd _build/html/``
4. open ``index.html`` in you internet browser

**NOTE: It is sometimes necessary to clean up _sphinx_ with the command ``make clean``, 
or/and delete the ``_autosummary`` folder to get rid of the warnings and error.**

### Adding a module
If we want to extend the *lmlib* with an additional module, for example called ``optim``,
we have to document the files in a certain way. This section gives an example how. 

* To create new modules (i.e. called ``your_module_name``) in the first layer (beside *poly*, *statespace*), do the flowing steps
    1. Generate a new folder in ``/lmlib/lmlib/`` an call it how the module should me named (here ``your_module_name``).
    2. create an ``__init__.py`` file in the ``your_module_name`` folder and type 
        ````
            
        """
        your_module_name
        ===================
        
        Description of your_module_name
        
        List of Submodules
        ------------------
        .. autosummary::
        :toctree: _your_module_name
        :template: module.rst
        
        your_file1_in_the_module
        your_file2_in_the_module
        
        """
        
        from your_file1_in_the_module import *
        from your_file2_in_the_module import *

        ````
    3. open the `module.rst` file in the _doc_ folder and ad your new module `your_module_name` 
    into the _autosummary_ list 
        ````rest
       List of Modules
       ===============
       
       lmlib contains following modules
       
       
       .. currentmodule:: lmlib
       
       .. autosummary::
          :toctree: _autosummary
       
          poly
          statespace
          your_module_name
        ````
    4. generate the documentation with `make html`

* Doc : TODO