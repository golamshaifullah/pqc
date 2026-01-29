Examples
========

Configured pipeline
-------------------

.. code-block:: python

   from pqc.pipeline import run_pipeline
   from pqc.config import BadMeasConfig, TransientConfig, MergeConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       backend_col="group",
       bad_cfg=BadMeasConfig(tau_corr_days=0.03, fdr_q=0.02),
       tr_cfg=TransientConfig(tau_rec_days=10.0, delta_chi2_thresh=30.0),
       merge_cfg=MergeConfig(tol_days=3.0 / 86400.0),
   )

Feature structure diagnostics
-----------------------------

.. code-block:: python

   from pqc.pipeline import run_pipeline
   from pqc.config import FeatureConfig, StructureConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       feature_cfg=FeatureConfig(add_orbital_phase=True, add_solar_elongation=True),
       struct_cfg=StructureConfig(
           mode="both",
           p_thresh=0.01,
           structure_group_cols=("group", "freq_bin"),
       ),
   )

Step and DM-step interpretation
-------------------------------

Step events model abrupt offsets in the residuals, while DM-step events use
frequency scaling consistent with dispersion:

.. math::

   r(t, f) = \frac{A}{f^2} H(t - t_0)

This is a direct consequence of cold-plasma dispersion and is standard in
pulsar timing analyses [LKH2005]_.

Plotting event membership
-------------------------

.. code-block:: python

   from pqc.utils.diagnostics import event_membership_mask

   # Informative membership (default)
   mask = event_membership_mask(df)

   # Applicable membership
   mask_app = event_membership_mask(df, use_applicable=True)

References
----------

.. [LKH2005] Lorimer, D. R., & Kramer, M. (2005). *Handbook of Pulsar Astronomy*.
   Cambridge University Press. citeturn1search3
