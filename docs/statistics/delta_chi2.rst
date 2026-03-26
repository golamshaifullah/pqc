Delta-Chi2 Model Comparison
===========================

What it is
----------

Several PQC event detectors (transient, step, DM-step, dip, bump, glitch)
accept candidates using improvement in weighted fit:

.. math::

   \Delta\chi^2 = \chi^2_{\mathrm{null}} - \chi^2_{\mathrm{model}}

Why PQC uses it
---------------

It provides a simple, fast statistic for ranking candidate event templates and
applying practical acceptance thresholds.

Weighted least-squares context
------------------------------

With residuals :math:`y_i`, uncertainties :math:`\sigma_i`, and model
:math:`m_i`, weights are :math:`w_i=1/\sigma_i^2`:

.. math::

   \chi^2_{\mathrm{model}} = \sum_i w_i (y_i - m_i)^2,\quad
   \chi^2_{\mathrm{null}} = \sum_i w_i y_i^2

Interpretation
--------------

- larger :math:`\Delta\chi^2` means stronger evidence for event model
- threshold choice controls detector aggressiveness

Assumptions and caveats
-----------------------

- assumes sigma values are meaningful relative weights
- not a full Bayesian model comparison
- scanning many candidate epochs/windows introduces a look-elsewhere effect,
  so absolute statistical significance is approximate

Small worked example
--------------------

If :math:`\chi^2_{\mathrm{null}}=120` and :math:`\chi^2_{\mathrm{model}}=92`,
then :math:`\Delta\chi^2=28`.
If detector threshold is ``25``, this candidate is accepted.

References
----------

.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574.
.. [LKH2005] Lorimer, D. R., & Kramer, M. (2005). *Handbook of Pulsar Astronomy*.
   Cambridge University Press.
