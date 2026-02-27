Command line
============

The CLI wraps the full pipeline and writes a CSV output table. Most options
map directly to configuration dataclasses in ``pqc.config``.

Note: advanced global-event tuning (solar/eclipses/gaussian-bumps/glitches)
is currently configured through the Python API config objects.

Basic run
---------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv

This writes a settings file alongside the CSV unless you override it with
``--settings-out``.

Custom thresholds
-----------------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --backend-col group \
     --tau-corr-min 45 \
     --fdr-q 0.02 \
     --tau-rec-days 10 \
     --delta-chi2 30

Exponential dips
----------------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --dip-tau-rec-days 30 \
     --dip-window-mult 5 \
     --dip-min-points 6 \
     --dip-delta-chi2 25 \
     --dip-member-eta 1.0 \
     --dip-scope backend

Logging
-------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --log-level DEBUG \
     --log-format "%(levelname)s:%(message)s"

Statistical knobs (what they mean)
----------------------------------

- ``--fdr-q`` sets the target false discovery rate for bad-measurement tests
  (Benjamini-Hochberg) [BH1995]_.
- ``--delta-chi2`` is the minimum improvement in step or transient likelihood
  required to accept a detection.
- ``--member-eta`` (step/transient configs) sets a per-point SNR threshold:

.. math::

   z_i = \frac{|m_i|}{\sigma_i} \quad \text{and require} \quad z_i > \eta

where ``m_i`` is the model effect at point ``i``.

Feature structure and grouping
------------------------------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --add-freq-bin --freq-bins 8 \
     --structure-mode both \
     --structure-group-cols group,freq_bin \
     --structure-test-features solar_elongation_deg,orbital_phase \
     --structure-detrend-features solar_elongation_deg,orbital_phase \
     --structure-summary-out structure_summary.csv

Preprocessing for selected detectors
------------------------------------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --detrend-features orbital_phase \
     --rescale-feature solar_elongation_deg \
     --condition-on group,freq_bin \
     --use-preproc-for ou,transient

Settings output
---------------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --settings-out results/run_settings.toml

References
----------

.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
   citeturn1search0
