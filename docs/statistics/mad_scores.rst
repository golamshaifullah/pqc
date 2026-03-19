MAD-Based Robust Scores
=======================

What it is
----------

PQC includes a robust outlier detector based on the median and median absolute
deviation (MAD), used to reduce sensitivity to a small fraction of extreme
points.

Definitions
-----------

For residuals :math:`y_i`:

.. math::

   \mathrm{med} = \mathrm{median}(y_i)

.. math::

   \mathrm{MAD} = \mathrm{median}(|y_i - \mathrm{med}|)

Robust z-score used in PQC:

.. math::

   z_i = 0.6745\frac{y_i-\mathrm{med}}{\mathrm{MAD}}

Why PQC uses it
---------------

Unlike mean/std z-scores, median/MAD remains stable when a subset of points is
contaminated, making it a useful complementary detector.

Interpretation
--------------

- larger :math:`|z_i|` indicates stronger deviation from robust center
- points are flagged if :math:`|z_i| \ge z_{\mathrm{thresh}}`

Assumptions and caveats
-----------------------

- majority of observations are not outliers
- MAD can be zero for near-constant or quantized data; then robust z-scores
  are undefined and detector may return no flags

Small worked example
--------------------

If ``median = 0`` and ``MAD = 2e-7``, a point with residual ``2e-6`` has:

.. math::

   z \approx 0.6745 \times 10 = 6.745

With threshold ``z_thresh = 5``, it is flagged.

References
----------

.. [Hampel1974] Hampel, F. R. (1974).
   "The influence curve and its role in robust estimation."
   *Journal of the American Statistical Association*, 69(346), 383-393.
.. [Rousseeuw1993] Rousseeuw, P. J., & Croux, C. (1993).
   "Alternatives to the median absolute deviation."
   *Journal of the American Statistical Association*, 88(424), 1273-1283.
