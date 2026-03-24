Pipeline Overview
=================

What this pipeline does
-----------------------

PQC performs quality control and event detection on pulsar timing residuals.
Given a pulsar timing model and TOAs, it:

1. merges timing arrays with metadata,
2. derives explanatory features,
3. runs bad-point and event detectors,
4. separates likely junk from coherent structure.

Core timing quantity
--------------------

Residuals are the observed-minus-modeled timing differences:

.. math::

   r_i = t_i^{\mathrm{obs}} - t_i^{\mathrm{model}}

Residuals should cluster around zero after a good model fit, but practical data
also include systematics, propagation effects, and real astrophysical events.

Input pairing and merge behavior
--------------------------------

PQC requires:

- ``X.par`` (timing model)
- sibling ``X_all.tim`` (TOAs; can include recursive ``INCLUDE`` lines)

The pipeline loads timing arrays via ``libstempo`` from ``.par + _all.tim``,
parses tim metadata/flags from the timfile tree, then merges by time/frequency
tolerances.

Why grouping matters
--------------------

Many detectors run per backend (for example ``backend_col="sys"``) because
different observing systems can have different offsets/noise properties.
Some detectors are global by design (for astrophysical hypotheses expected to
appear across systems).

References
----------

.. [Hobbs2006] Hobbs, G. B., Edwards, R. T., & Manchester, R. N. (2006).
   "tempo2, a new pulsar-timing package - I. An overview." *MNRAS*, 369(2), 655-672.
.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574.
