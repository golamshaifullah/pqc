Detector Stack and Configuration Logic
======================================

Layered view
------------

PQC is easiest to reason about in layers:

- **Layer A (features/merge):** metadata merge and explanatory features
  (orbital phase, solar elongation, elevation, airmass, parallactic angle,
  frequency bins).
- **Layer B (generic QC):** bad-point detection and robust outlier checks.
- **Layer C (event models):** transient, dip, step, DM-step, solar, eclipse,
  gaussian-bump, glitch.
- **Layer D (interpretation):** event-aware semantics and final flags.

Backend-specific vs global detectors
------------------------------------

Backend-specific detectors are best for instrument-local effects.
Global detectors are best for astrophysical/line-of-sight hypotheses.

Typical mapping:

- often backend-aware: transient, step, robust outlier
- often global: dip, solar, bump, glitch, eclipse (if enabled)

Event families in plain language
--------------------------------

- **Transient**: short-lived exponential recovery after onset.
- **Dip**: fast drop with longer recovery, optionally chromatic
  :math:`1/f^\alpha`.
- **Step**: achromatic level jump after ``t0``.
- **DM-step**: chromatic step with near-:math:`1/f^2` scaling.
- **Solar event**: elongation-linked behavior near the Sun, optionally fitted
  per-year with global fallback.
- **Gaussian bump**: broad hump-like MJD event over multiple trial durations.
- **Glitch**: long-timescale step/ramp or peak+ramp behavior.

How thresholds work together
----------------------------

- ``min_points`` prevents sparse/noisy coincidences.
- ``delta_chi2_thresh`` requires meaningful fit improvement.
- ``member_eta`` controls which TOAs become event members after a candidate is
  accepted.

Worked membership example
-------------------------

If model effect at a TOA is ``2 us`` and sigma is ``1.5 us``:

.. math::

   |m|/\sigma = 2/1.5 \approx 1.33

For ``member_eta = 1``, this TOA is a member; for ``member_eta = 2``, it is not.

References
----------

.. [LKH2005] Lorimer, D. R., & Kramer, M. (2005). *Handbook of Pulsar Astronomy*.
   Cambridge University Press.
.. [Edwards2006] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006).
   "tempo2, a new pulsar timing package - II. The timing model and precision estimates."
   *MNRAS*, 372(4), 1549-1574.
