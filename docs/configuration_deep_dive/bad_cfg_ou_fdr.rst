Bad-Point Detector (OU + FDR)
=============================

Problem this stage solves
-------------------------

The bad-point detector asks: which TOAs are implausible after accounting for
short-timescale correlation between nearby residuals?

If correlation is ignored, clustered fluctuations are often over-flagged.

OU model in one line
--------------------

PQC uses an Ornstein-Uhlenbeck (OU) process as a short-memory mean-reverting
noise baseline:

.. math::

   dX_t = -\lambda X_t\,dt + \sigma\,dW_t

with correlation timescale :math:`\tau = 1/\lambda`.

Operational meaning of ``tau_corr_days``
----------------------------------------

If ``tau_corr_days ≈ 0.0208`` (about 30 minutes), the lag correlation is:

.. math::

   \mathrm{Corr}(\Delta) \approx e^{-|\Delta|/\tau}

Example:

- 15 min lag: :math:`e^{-0.5} \approx 0.61`
- 30 min lag: :math:`e^{-1} \approx 0.37`
- 60 min lag: :math:`e^{-2} \approx 0.14`

So minute-scale neighbors are not treated as independent.

Why FDR is used
---------------

Multiple days are tested, so raw significance is not enough. PQC applies
Benjamini-Hochberg false discovery rate (FDR) control at ``fdr_q``.

FDR controls expected false-positive fraction among flagged detections:

.. math::

   \mathrm{FDR} = E\left[\frac{V}{\max(R,1)}\right]

where :math:`V` is false positives and :math:`R` is total flagged.

Meaning of ``mark_only_worst_per_day``
--------------------------------------

- ``true``: mark only the single most extreme TOA on each flagged day.
- ``false``: allow multiple TOAs on a flagged day to be marked.

This is useful when an observing session is broadly compromised.

Worked BH example
-----------------

Sorted p-values: ``[0.001, 0.006, 0.011, 0.030, 0.200]``, target ``q=0.02``.
BH thresholds are ``[0.004, 0.008, 0.012, 0.016, 0.020]``. First three pass,
so three hypotheses are rejected.

References
----------

.. [UO1930] Uhlenbeck, G. E., & Ornstein, L. S. (1930).
   "On the theory of the Brownian motion." *Physical Review*, 36, 823-841.
.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
