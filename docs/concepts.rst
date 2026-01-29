Concepts
========

Backends and grouping
---------------------

Most detectors operate per backend (column ``group`` by default). This keeps
instrument-specific noise characteristics separate. You can select the backend
column with ``backend_col`` in the pipeline or ``--backend-col`` in the CLI.

Residuals and noise
-------------------

Pulsar timing uses residuals between observed and modeled TOAs. In many
analyses, residuals are treated as approximately Gaussian with known
uncertainties, so weights scale as ``1/sigma^2``. The general framework is
standard in timing packages such as tempo2 [Hobbs2006]_ [Edwards2006]_.

Bad measurement detection
-------------------------

The OU-based bad measurement detector models short-timescale residual
structure with an Ornstein-Uhlenbeck (OU) process, a mean-reverting Gaussian
process defined by the stochastic differential equation:

.. math::

   dX_t = -\frac{1}{\tau} X_t\, dt + \sigma\, dW_t

Outliers are identified via innovation statistics and multiple-testing control.
The Benjamini-Hochberg procedure controls the false discovery rate (FDR) by
comparing ordered p-values to a linearly increasing threshold [BH1995]_.

Transient events
----------------

Transient exponential recoveries are modeled as

.. math::

   r(t) = A\, e^{-(t - t_0)/\tau}\, H(t - t_0)

where ``H`` is the Heaviside step function, ``t0`` is the event time, ``A`` is
amplitude, and ``tau`` is the recovery timescale. This is a common template for
recovery-like disturbances in timing data [LKH2005]_.

Step and DM-step events
-----------------------

Step events model abrupt offsets at time ``t0``:

.. math::

   r(t) = A\, H(t - t_0)

DM-step events scale with observing frequency ``f`` according to the
dispersion delay law:

.. math::

   r(t, f) = \frac{A}{f^2}\, H(t - t_0)

The ``1/f^2`` scaling reflects cold-plasma dispersion and is standard in
pulsar timing [LKH2005]_.

Membership semantics
--------------------

Step and DM-step events use two masks:

- ``*_applicable``: points in the time window (``t >= t0`` and within
  the membership window).
- ``*_informative``: applicable points that also satisfy the per-point
  SNR threshold (``|model_effect| / sigma > eta``).

Event identity is carried in ``step_id`` and ``dm_step_id`` and is based on
applicability. Plotting and summary defaults use informative membership.

Covariate-conditioned preprocessing
-----------------------------------

Detectors can operate on residuals that are detrended and/or variance-rescaled
as a function of covariates. This leaves the original residuals intact while
adding processed columns for selected detectors.

References
----------

.. [Hobbs2006] Hobbs, G. B., Edwards, R. T., & Manchester, R. N. (2006).
   "tempo2, a new pulsar-timing package - I. An overview." *MNRAS*, 369(2), 655-672.
   citeturn0search1
.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574. citeturn0search0
.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
   citeturn1search0
.. [UO1930] Uhlenbeck, G. E., & Ornstein, L. S. (1930).
   "On the theory of the Brownian motion." *Physical Review*, 36, 823-841.
   citeturn2search0
.. [LKH2005] Lorimer, D. R., & Kramer, M. (2005). *Handbook of Pulsar Astronomy*.
   Cambridge University Press. citeturn1search3
