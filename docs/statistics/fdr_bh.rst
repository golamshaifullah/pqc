Benjamini-Hochberg FDR
======================

What it is
----------

PQC uses Benjamini-Hochberg (BH) false discovery rate control on day-level
test p-values derived from residual anomaly scores.

FDR definition
--------------

False discovery rate is:

.. math::

   \mathrm{FDR} = \mathbb{E}\left[\frac{V}{\max(R,1)}\right]

where :math:`V` is the number of false positives and :math:`R` is the number
of rejected hypotheses.

BH procedure (used by PQC)
--------------------------

For sorted p-values :math:`p_{(1)} \le \cdots \le p_{(m)}` and target
:math:`q`:

1. find largest :math:`k` such that :math:`p_{(k)} \le (k/m)q`
2. reject all hypotheses with :math:`p \le p_{(k)}`

Why PQC uses it
---------------

Outlier scans test many days. BH controls expected false discoveries while
retaining more power than family-wise error controls (e.g., Bonferroni).

Assumptions and caveats
-----------------------

- BH guarantees hold under independence or certain positive dependence.
- If p-values are miscalibrated (model mismatch), realized FDR can drift.
- Day-level aggregation can miss finer within-day effects.

Small worked example
--------------------

Suppose ``q=0.02`` and sorted p-values are:
``[0.001, 0.012, 0.08, 0.2]``.
Thresholds are ``[0.005, 0.01, 0.015, 0.02]``.
Only the first p-value passes, so one day is flagged.

References
----------

.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
