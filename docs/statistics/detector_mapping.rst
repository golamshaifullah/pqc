Detector-to-Statistic Mapping
=============================

This page maps each PQC detector to its core statistical test(s), scoring
rules, and key assumptions.

Summary table
-------------

+--------------------------+------------------------------+-----------------------------+---------------------------+
| Detector                 | Primary statistic/model      | Typical threshold controls  | Key assumptions           |
+==========================+==============================+=============================+===========================+
| Bad measurements         | OU innovations + BH-FDR      | ``tau_corr_days``, ``fdr_q``| OU-like short-lag noise,  |
|                          | on day-level p-values        |                             | calibrated p-values       |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Robust outliers          | Median/MAD robust z-score    | ``z_thresh``                | majority inliers, MAD>0   |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Transients               | Exponential template +       | ``delta_chi2_thresh``,      | single dominant           |
|                          | :math:`\Delta\chi^2`         | ``member_eta``              | recovery in window        |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Step                     | Two-segment weighted means   | ``delta_chi2_thresh``,      | one changepoint,          |
|                          | + :math:`\Delta\chi^2`       | ``member_eta``              | achromatic offset         |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| DM-step                  | Step in DM-scaled space      | ``delta_chi2_thresh``,      | :math:`1/f^2` scaling     |
|                          | + :math:`\Delta\chi^2`       | ``member_eta``              | is appropriate            |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Exponential dip          | Dip/recovery template +      | ``delta_chi2_thresh``,      | dip-like morphology,      |
|                          | :math:`\Delta\chi^2`         | ``member_eta``              | optional :math:`1/f^\alpha`|
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Solar events             | Exponential vs elongation,   | ``approach_max_deg``,       | shape in elongation,      |
|                          | per-year/global fitting      | ``member_eta``              | optional :math:`1/f^\alpha`|
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Eclipse events           | Phase-centered template      | ``width_min/max``,          | binary phase available,   |
|                          | + :math:`\Delta\chi^2`       | ``member_eta``              | eclipse-centered shape    |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Gaussian-bump            | Multi-model bump comparison  | ``delta_chi2_thresh``,      | event resembles tested    |
|                          | (gaussian/laplace/plateau)   | ``member_eta``              | bump templates            |
+--------------------------+------------------------------+-----------------------------+---------------------------+
| Glitch                   | Step+ramp or peak+ramp       | ``delta_chi2_thresh``,      | post-glitch trend shape,  |
|                          | model comparison             | ``noise_k``                 | long-duration behavior    |
+--------------------------+------------------------------+-----------------------------+---------------------------+

Detailed mapping notes
----------------------

Bad measurements (OU + FDR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Build innovation z-scores under OU correlation.
2. Aggregate to day-level maxima of :math:`|z|`.
3. Convert to p-values and apply BH-FDR.

This controls multiplicity across many tested days while accounting for short
time correlation.

Robust outliers (MAD)
~~~~~~~~~~~~~~~~~~~~~

Uses robust standardized residuals:

.. math::

   z_i = 0.6745\frac{y_i-\mathrm{median}(y)}{\mathrm{MAD}}

Flags points with :math:`|z_i| \ge z_\mathrm{thresh}`.

Event detectors (:math:`\Delta\chi^2` family)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most event detectors compare a null and event model in weighted least squares:

.. math::

   \Delta\chi^2 = \chi^2_{\mathrm{null}} - \chi^2_{\mathrm{model}}

Accepted events exceed configured ``delta_chi2_thresh`` and then apply
membership rules based on per-point model SNR (``member_eta``).

Frequency-dependent detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several detectors support chromatic scaling:

.. math::

   m(f) \propto \frac{1}{f^\alpha}

where :math:`\alpha` may be fixed or fitted in configured bounds.
DM-step uses the physically motivated :math:`\alpha=2`.

Common caveats across detectors
-------------------------------

- ``sigma`` quality strongly impacts weighted statistics.
- scanning many candidate epochs/windows induces look-elsewhere effects.
- model mismatch can turn real structure into apparent outliers (or vice versa).
- event precedence and overlap suppression settings affect final labels.

References
----------

.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
.. [UO1930] Uhlenbeck, G. E., & Ornstein, L. S. (1930).
   "On the theory of the Brownian motion." *Physical Review*, 36, 823-841.
.. [Hampel1974] Hampel, F. R. (1974).
   "The influence curve and its role in robust estimation."
   *Journal of the American Statistical Association*, 69(346), 383-393.
.. [Rousseeuw1993] Rousseeuw, P. J., & Croux, C. (1993).
   "Alternatives to the median absolute deviation."
   *Journal of the American Statistical Association*, 88(424), 1273-1283.
.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574.
