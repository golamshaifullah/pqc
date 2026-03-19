OU Innovations
==============

What it is
----------

PQC models short-timescale correlated residual noise with an
Ornstein-Uhlenbeck (OU) process. The OU process is a mean-reverting Gaussian
stochastic process:

.. math::

   dX_t = -\frac{1}{\tau}X_t\,dt + \sigma\,dW_t

Why PQC uses it
---------------

TOA residuals are often not independent at short lags. OU innovations provide
a local whitening transform so pointwise surprises are measured against a
correlated-noise baseline rather than white-noise assumptions.

Discrete innovation form used in PQC
------------------------------------

For irregular samples at times :math:`t_i`:

.. math::

   \phi_i = \exp\left(-\frac{t_i-t_{i-1}}{\tau}\right)

.. math::

   e_i = y_i - \phi_i y_{i-1}

.. math::

   v_i = \sigma_i^2 + \phi_i^2 \sigma_{i-1}^2 + q(1-\phi_i^2)

.. math::

   z_i = \frac{e_i}{\sqrt{v_i}}

where ``q`` is an additional variance term estimated robustly.

Interpretation
--------------

- :math:`|z_i| \approx 0`: consistent with the OU-noise prediction.
- large :math:`|z_i|`: locally inconsistent, candidate bad measurement.

Assumptions and caveats
-----------------------

- OU is only an approximation of residual correlation.
- Innovation tails are treated approximately Gaussian.
- Non-monotonic timestamps or severe model mismatch can distort scores.

Small worked example
--------------------

If :math:`\tau=10` d and :math:`\Delta t=1` d, then
:math:`\phi=\exp(-0.1)=0.905`.
With :math:`y_{i-1}=1.0`, :math:`y_i=0.2`:
:math:`e_i = 0.2 - 0.905 = -0.705`.
If :math:`v_i=0.25`, then :math:`z_i=-1.41`.

References
----------

.. [UO1930] Uhlenbeck, G. E., & Ornstein, L. S. (1930).
   "On the theory of the Brownian motion." *Physical Review*, 36, 823-841.
.. [Gardiner2009] Gardiner, C. (2009). *Stochastic Methods* (4th ed.).
   Springer.
