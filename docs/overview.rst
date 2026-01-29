Overview
========

PQC is a lightweight quality-control toolkit for pulsar timing array (PTA)
residuals. It parses tempo2 timfiles, loads TOA/residual arrays via libstempo,
merges timing arrays with timfile metadata, normalizes backend keys, and
annotates each TOA with QC flags and event detections. The scientific context
is standard pulsar timing: compare observed pulse times of arrival (TOAs) with
a timing model to obtain residuals, then search for structure and outliers in
those residuals [Hobbs2006]_ [Edwards2006]_ [Vigeland2014]_ [LKH2005]_.

Core capabilities
-----------------

- Parse tempo2 timfiles (including INCLUDE recursion and flags)
- Load TOA/residual arrays via libstempo
- Merge timing arrays with timfile metadata
- Normalize backend keys (sys/group) for per-backend analysis
- Detect bad measurements and transient exponential recoveries
- Detect step and DM-step offsets
- Optional feature columns and structure diagnostics
- Optional covariate-conditioned preprocessing

Why a QC layer?
--------------

Pulsar timing residuals are used in precision measurements and PTA analyses.
Small deviations in instrument behavior or observing conditions can produce
outliers or structured residuals that bias inference if not flagged. PQC
provides a reproducible, pipeline-style QC stage prior to scientific analysis,
while remaining lightweight and transparent.

PTA context
-----------

PTA experiments rely on correlated timing residuals across many pulsars. In
this setting, robust QC and consistent residual diagnostics are critical
inputs to downstream analyses [Hobbs2010]_.

References
----------

.. [Hobbs2006] Hobbs, G. B., Edwards, R. T., & Manchester, R. N. (2006).
   "tempo2, a new pulsar-timing package - I. An overview." *MNRAS*, 369(2), 655-672.
   citeturn0search1
.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574. citeturn0search0
.. [Vigeland2014] Vigeland, S. J., & Vallisneri, M. (2014).
   "Bayesian inference for pulsar-timing models." *MNRAS*, 440(2), 1446-1457.
   citeturn1search1
.. [LKH2005] Lorimer, D. R., & Kramer, M. (2005). *Handbook of Pulsar Astronomy*.
   Cambridge University Press. citeturn1search3
.. [Hobbs2010] Hobbs, G., et al. (2010).
   "The International Pulsar Timing Array project: using pulsars as a gravitational wave detector."
   *Classical and Quantum Gravity*, 27(8), 084013. citeturn0search3
