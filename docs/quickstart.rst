Quickstart
==========

Before you start
----------------

PQC expects a ``.par`` file and a sibling ``*_all.tim`` file produced by
tempo2. Example: ``J1909-3744.par`` and ``J1909-3744_all.tim`` in the same
directory.

CLI (fastest path)
------------------

.. code-block:: bash

   pqc --par /path/to/J1909-3744.par --out results/J1909-3744_qc.csv

This writes:

- the QC CSV output to ``results/J1909-3744_qc.csv``
- the run settings TOML to ``results/J1909-3744_qc.pqc_settings.toml``

Python (notebook-friendly)
--------------------------

.. code-block:: python

   import pandas as pd
   from pqc.pipeline import run_pipeline

   df = run_pipeline("/path/to/J1909-3744.par")

   # Quick sanity checks
   print(df.columns)
   print(df[["mjd", "resid", "sigma", "bad_point", "event_member"]].head())

Inspect results
--------------

.. code-block:: python

   # Keep only good points
   good = df.loc[~df["bad_point"].fillna(False)].copy()

   # Focus on event members (transients, steps, dips, solar/eclipses, etc.)
   events = df.loc[df["event_member"].fillna(False)].copy()

   # Save a filtered table
   good.to_csv("results/J1909-3744_qc_good.csv", index=False)

What you get
------------

The pipeline returns a pandas DataFrame with timing metadata, QC flags, and
optional event annotations. Key columns include:

- ``bad_point``: aggregate bad measurement flag
- ``event_member``: aggregate event membership across enabled detectors
- ``step_applicable`` and ``step_informative``
- ``dm_step_applicable`` and ``dm_step_informative``
- event-specific columns such as ``exp_dip_member``, ``solar_event_member``,
  ``eclipse_event_member``, ``gaussian_bump_member``, and ``glitch_member``

Important semantics
-------------------

Rows marked as astrophysical-event members are treated as non-outliers in
``bad_point``. The compatibility field ``outlier_any`` remains available and is
defined as ``bad_point OR event_member``.

If you run via the CLI, a TOML settings file is also written alongside the
CSV (same filename stem with ``.pqc_settings.toml``). This captures the exact
configuration used for the run.

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
