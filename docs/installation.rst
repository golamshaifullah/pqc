Installation
============

PQC is a pure-Python package, but running the pipeline requires ``libstempo``
and a working tempo2 installation for pulsar timing residuals.

Editable install (development)
------------------------------

.. code-block:: bash

   pip install -e .

Install with libstempo support (for running the full pipeline):

.. code-block:: bash

   pip install -e ".[libstempo]"

Standard install from a local checkout
--------------------------------------

.. code-block:: bash

   pip install .

Dependencies and environment
----------------------------

- ``libstempo`` is required for loading TOAs and residuals in the full
  pipeline.
- tempo2 must be installed and configured for ``libstempo`` to work.
- The documentation can be built without tempo2 because the docs mock
  ``libstempo`` during autodoc.

If you only want to read the docs or inspect the API, the build will complete
without a working tempo2 installation. For actual data processing, verify that
``libstempo`` can load your par/tim files.
