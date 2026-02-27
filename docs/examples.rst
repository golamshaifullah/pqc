Examples
========

Tutorial: end-to-end QC in 5 minutes
------------------------------------

1) Run PQC and save output:

.. code-block:: bash

   pqc --par /path/to/J1909-3744.par --out results/J1909-3744_qc.csv

2) Load the results:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("results/J1909-3744_qc.csv")

3) See the key flags:

.. code-block:: python

   df[["bad_point", "event_member"]].mean()

4) Export just the clean subset:

.. code-block:: python

   clean = df.loc[~df["bad_point"].fillna(False)].copy()
   clean.to_csv("results/J1909-3744_clean.csv", index=False)

5) Check the settings used:

.. code-block:: python

   # settings file is written alongside the CSV
   with open("results/J1909-3744_qc.pqc_settings.toml") as f:
       print(f.read().splitlines()[:20])

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
   import matplotlib.pyplot as plt

   # Informative membership (default)
   mask = event_membership_mask(df)

   # Applicable membership
   mask_app = event_membership_mask(df, use_applicable=True)

   plt.scatter(df["mjd"], df["resid"], s=6, alpha=0.5, label="all")
   plt.scatter(df.loc[mask, "mjd"], df.loc[mask, "resid"], s=8, label="event members")
   plt.xlabel("MJD")
   plt.ylabel("Residual (s)")
   plt.legend()

Solar elongation diagnostic
---------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   if "solar_elongation_deg" in df.columns:
       plt.scatter(df["solar_elongation_deg"], df["resid"], s=6, alpha=0.5)
       plt.xlabel("Solar elongation (deg)")
       plt.ylabel("Residual (s)")
       plt.title("Residuals vs. solar elongation")

Preprocessing before detectors
------------------------------

.. code-block:: python

   from pqc.config import PreprocConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       preproc_cfg=PreprocConfig(
           detrend_features=("orbital_phase",),
           rescale_feature="solar_elongation_deg",
           condition_on=("group", "freq_bin"),
           use_preproc_for=("ou", "transient"),
       ),
   )

Hard sigma gate
---------------

.. code-block:: python

   from pqc.config import OutlierGateConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       gate_cfg=OutlierGateConfig(enabled=True, sigma_thresh=3.0),
   )

Exponential dip events
----------------------

.. code-block:: python

   from pqc.config import ExpDipConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       dip_cfg=ExpDipConfig(
           tau_rec_days=30.0,
           window_mult=5.0,
           min_points=6,
           delta_chi2_thresh=25.0,
           member_eta=1.0,
       ),
   )

   dip_members = df.loc[df["exp_dip_member"].fillna(False)]

Solar and eclipse events
------------------------

.. code-block:: python

   from pqc.config import EclipseConfig, SolarCutConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       solar_cfg=SolarCutConfig(
           enabled=True,
           freq_dependence=True,
           min_points_global=20,
           min_points_year=8,
       ),
       eclipse_cfg=EclipseConfig(
           enabled=True,
           center_phase=0.25,
           freq_dependence=True,
       ),
   )

   solar_members = df.loc[df["solar_event_member"].fillna(False)]
   eclipse_members = df.loc[df["eclipse_event_member"].fillna(False)]

Gaussian-bumps and glitches
---------------------------

.. code-block:: python

   from pqc.config import GaussianBumpConfig, GlitchConfig

   df = run_pipeline(
       "/path/to/pulsar.par",
       bump_cfg=GaussianBumpConfig(enabled=True),
       glitch_cfg=GlitchConfig(enabled=True, min_duration_days=1000.0),
   )

   bumps = df.loc[df["gaussian_bump_member"].fillna(False)]
   glitches = df.loc[df["glitch_member"].fillna(False)]

References
----------

.. [LKH2005] Lorimer, D. R., & Kramer, M. (2005). *Handbook of Pulsar Astronomy*.
   Cambridge University Press. citeturn1search3
