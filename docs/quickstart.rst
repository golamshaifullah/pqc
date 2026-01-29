Quickstart
==========

CLI
---

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv

Python
------

.. code-block:: python

   from pqc.pipeline import run_pipeline

   df = run_pipeline("/path/to/pulsar.par")

What you get
------------

The pipeline returns a pandas DataFrame with timing metadata, QC flags, and
optional event annotations. Key columns include:

- ``bad_point``: aggregate bad measurement flag
- ``event_member``: event membership (transient, informative step, informative DM-step)
- ``step_applicable`` and ``step_informative``
- ``dm_step_applicable`` and ``dm_step_informative``

Minimal statistical context
---------------------------

Residuals are the difference between observed and modeled TOAs:

.. math::

   R_i = t_i^{\mathrm{obs}} - t_i^{\mathrm{model}}

Many timing analyses assume (or approximate) independent Gaussian errors, so
weights are often proportional to ``1/sigma_i^2`` and the usual chi-square
statistic appears:

.. math::

   \chi^2 = \sum_i \left(\frac{R_i}{\sigma_i}\right)^2

PQC uses these same residuals and uncertainties but focuses on QC tasks such as
outlier detection and event membership labeling rather than full model fitting
[Edwards2006]_ [Vigeland2014]_.

References
----------

.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574. citeturn0search0
.. [Vigeland2014] Vigeland, S. J., & Vallisneri, M. (2014).
   "Bayesian inference for pulsar-timing models." *MNRAS*, 440(2), 1446-1457.
   citeturn1search1
