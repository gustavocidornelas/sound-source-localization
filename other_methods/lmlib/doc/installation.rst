.. _lmlib_installation:

Installation
============

Installation via ``pip``
------------------------
Make sure you have installed ``pip`` and then type into the console/bash:

.. code-block:: bash

   $ pip install lmlib

`lmlib` gets downloaded from https://pypi.org/project/lmLib/ and installed into current your python environment.

If you have installed ``lmlib``, the :ref:`lmlib_getting_started` helps to get used to work with lmlib.

Installation via VCS (for developers)
-------------------------------------
Clone the project from ``gitlab`` and make sure you have ``pip`` already installed.

.. code-block:: bash

   $ git clone https://gitlab.ti.bfh.ch/lmlib/lmlib
   $ cd lmlib
   $ pip install -r requirements.txt
   $ pip install -e .