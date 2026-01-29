Command line
============

The CLI wraps the full pipeline and writes a CSV output table. Most options
map directly to configuration dataclasses in ``pqc.config``.

Basic run
---------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv

Custom thresholds
-----------------

.. code-block:: bash

   pqc --par /path/to/pulsar.par --out out.csv \
     --backend-col group \
     --tau-corr-min 45 \
     --fdr-q 0.02 \
     --tau-rec-days 10 \
     --delta-chi2 30

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

References
----------

.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
   citeturn1search0
