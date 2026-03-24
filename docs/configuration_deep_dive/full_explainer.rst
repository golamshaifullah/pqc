Expanded standalone explanation of the PQC configuration
========================================================

1. What this pipeline is for
----------------------------

This pipeline performs **quality control** and **event detection** on pulsar timing data.

In pulsar timing, the basic observable is a set of **TOAs** — times of arrival of pulses measured at the telescope. A timing package such as TEMPO2 converts those measurements into barycentric arrival times, compares them with a physical timing model, and computes **timing residuals**, which are the differences between what was observed and what the model predicted. In a perfectly modeled world, residuals would be consistent with zero apart from noise. In practice, residuals also contain instrumental systematics, propagation effects, and sometimes genuine astrophysical events. [1]_

A compact way to write this is:

.. math::

   r_i = t^{\mathrm{obs}}_i - t^{\mathrm{model}}_i,

where :math:`r_i` is the residual for TOA :math:`i`.

So the job of the pipeline is to:

1. decide which points are probably bad measurements,
2. decide which unusual patterns are probably real structured events,
3. separate instrumental effects from astrophysical ones,
4. avoid confusing “interesting” with “garbage”.

That distinction matters. A point can be unusual because it belongs to a real event, not because it is junk.

2. What the run configuration means
-----------------------------------

``run``
~~~~~~~

The ``parfile`` is the pulsar timing model used as the reference model. The pipeline automatically pairs it with the sibling ``*_all.tim`` file containing the TOAs.

The setting ``backend_col="sys"`` means observations are grouped by the ``sys`` tag, which usually identifies the observing backend, data-taking chain, or system label. That is standard practice in pulsar timing, because different telescopes, receivers, and backends often carry different offsets or noise properties; TEMPO2 explicitly supports fitting jumps or grouping TOAs by this kind of metadata. [1]_

The setting ``drop_unmatched=false`` means TOAs are **kept** even if the metadata merge does not find a perfect match. This is a conservative data-retention choice: do not throw points away early just because the bookkeeping was imperfect.

3. The logic of the whole detector stack
----------------------------------------

The stack is easiest to understand if you think of it in layers.

**Layer A: bookkeeping and feature generation**

* merge TOAs with metadata,
* derive useful explanatory features such as orbital phase or solar elongation.

**Layer B: generic quality control**

* look for bad points,
* look for extreme outliers.

**Layer C: structured event models**

* look for transients, dips, steps, solar effects, bumps, and glitches.

**Layer D: interpretation**

* if a point belongs to a coherent event model, treat it differently from an isolated bad point.

That last rule is crucial: a measurement that participates in a well-fit event is not handled in the same way as a lone absurd datapoint.

4. Bad-point detector: OU model, FDR control, and per-day marking
-----------------------------------------------------------------

4.1 What problem this detector is solving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bad-point detector asks:

   “Which observations look implausible even after accounting for the fact that nearby observations can be correlated?”

That second clause is the key point. If you assume every residual is independent, then any short run of elevated points looks suspicious. But real timing data often have **short-memory correlation**: residuals close in time tend to move together. If you ignore that, you over-flag.

So the detector uses an **Ornstein–Uhlenbeck process** as its noise baseline.

4.2 What an Ornstein–Uhlenbeck process is
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An **Ornstein–Uhlenbeck (OU) process** is a standard continuous-time stochastic model for **mean-reverting correlated noise**.

A common form is:

.. math::

   dX_t = -\lambda X_t\,dt + \sigma\, dW_t,

where:

* :math:`X_t` is the noise process,
* :math:`\lambda > 0` controls how quickly it relaxes back toward zero,
* :math:`\sigma` sets the noise strength,
* :math:`W_t` is Brownian motion.

There are two main ideas in that equation.

**Mean reversion**

The term :math:`-\lambda X_t\,dt` pulls the process back toward zero. If :math:`X_t` is positive, the drift is negative. If :math:`X_t` is negative, the drift is positive.

That is why OU is called “mean-reverting”: excursions do not simply wander off indefinitely; they tend to relax.

**Short-memory correlation**

An OU process is not white noise. Nearby times are correlated. In the stationary case, the correlation falls off exponentially with lag:

.. math::

   \mathrm{Corr}(X_t, X_{t+\Delta}) = e^{-|\Delta|/\tau},

where

.. math::

   \tau = \frac{1}{\lambda}

is the correlation timescale.

That exponential falloff is the key operational fact for this configuration.

4.3 What ``tau_corr_days ≈ 0.0208`` means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This setting corresponds to:

.. math::

   \tau \approx 0.0208 \text{ days} \approx 30 \text{ minutes}.

So the detector assumes that residuals measured within roughly half an hour can still be meaningfully correlated.

**Worked example**

Take three time separations:

* **15 minutes** = :math:`0.0104` days

  .. math::

     e^{-0.0104/0.0208} = e^{-0.5} \approx 0.61

  So there is still fairly strong correlation.

* **30 minutes** = :math:`0.0208` days

  .. math::

     e^{-1} \approx 0.37

  The correlation is weaker, but still present.

* **60 minutes** = :math:`0.0417` days

  .. math::

     e^{-2} \approx 0.14

  At that point the dependence is small.

So, in plain English: points separated by minutes are not treated as independent coin flips; points separated by an hour are close to independent.

That is why this model is better than a pure white-noise model for flagging bad TOAs.

4.4 Why this matters for bad-point detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have four TOAs taken within 20 minutes, and all four residuals are somewhat high. Under an independent-noise model, you might say:

   “Four elevated points in a row — that looks very unlikely.”

But under an OU model, the correct answer may instead be:

   “Not that unlikely; once one point is high, nearby points are more likely to be high too.”

So the detector evaluates each point against a **correlated-noise baseline**, not an independence assumption. That reduces false alarms caused by short-timescale systematics or weather/instrument effects that affect adjacent TOAs together.

4.5 What FDR is
~~~~~~~~~~~~~~~

``fdr_q = 0.02`` means the detector controls the **False Discovery Rate** at 2%.

FDR is a multiple-testing concept introduced by Benjamini and Hochberg. The basic problem is that when you test many points, some will look significant just by chance. Instead of demanding the stronger condition “almost no false positives anywhere,” FDR controls the expected fraction of false positives **among the flagged set**. [2]_

Formally, if:

* :math:`R` = number of flagged points,
* :math:`V` = number of false positives among them,

then the false discovery rate is:

.. math::

   \mathrm{FDR} = E\!\left[\frac{V}{\max(R,1)}\right].

So ``fdr_q = 0.02`` means the procedure is tuned so that, on average, the flagged collection should contain only a small fraction of false alarms.

That does **not** mean every individual flagged point has exactly a 98% chance of being real. It is a property of the overall selection rule, not a per-point posterior probability.

4.6 Brief derivation of the Benjamini–Hochberg logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classic BH procedure works as follows. Suppose you have :math:`m` p-values:

.. math::

   p_{(1)} \le p_{(2)} \le \cdots \le p_{(m)}.

Pick the largest :math:`k` such that:

.. math::

   p_{(k)} \le \frac{k}{m} q,

where :math:`q` is the target FDR level.

Then reject all hypotheses from :math:`1` through :math:`k`.

The threshold gets looser as :math:`k` grows, which is why BH is less brutally conservative than Bonferroni while still controlling the overall false-discovery fraction under standard assumptions. [2]_

**Worked example**

Suppose there are 5 candidate bad points and the sorted p-values are:

.. math::

   0.001,\ 0.006,\ 0.011,\ 0.030,\ 0.200

with :math:`q=0.02`.

The BH thresholds are:

.. math::

   \frac{1}{5}0.02=0.004,\quad
   \frac{2}{5}0.02=0.008,\quad
   \frac{3}{5}0.02=0.012,\quad
   \frac{4}{5}0.02=0.016,\quad
   \frac{5}{5}0.02=0.020.

Compare:

* :math:`0.001 \le 0.004`: pass
* :math:`0.006 \le 0.008`: pass
* :math:`0.011 \le 0.012`: pass
* :math:`0.030 \nleq 0.016`: fail
* :math:`0.200 \nleq 0.020`: fail

So the first three would be flagged.

That is the sense in which FDR turns a pile of raw significances into a controlled list of detections.

4.7 What ``mark_only_worst_per_day = false`` means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This setting controls how the detector behaves once a day looks contaminated.

If it were ``true``, the pipeline would keep only the single worst TOA on a bad day.

Because it is ``false``, the pipeline can mark **all** TOAs on that day that satisfy the badness criterion.

**Worked example**

Imagine one observing day contains six TOAs. Their normalized residual significances are:

.. math::

   0.3,\ 0.7,\ 3.2,\ 4.8,\ 4.1,\ 0.5.

If that day is identified as problematic:

* with ``true``, you might mark only the :math:`4.8\sigma` point;
* with ``false``, you could mark the :math:`3.2\sigma`, :math:`4.8\sigma`, and :math:`4.1\sigma` points.

That is more permissive, and it makes sense when an entire observing session may have been compromised.

4.8 Bottom line for ``bad_cfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This config is saying:

* model residuals with **short-memory mean-reverting noise**,
* assume correlations survive on a **30-minute timescale**,
* use **strict multiple-testing control** at **2% FDR**,
* and if a day is bad, allow **multiple points from that day** to be marked.

This is a thoughtful detector design. It is trying hard not to confuse clustered noise with isolated catastrophic points.

5. Transient exponential-recovery detector (``tr_cfg``)
-------------------------------------------------------

5.1 What it looks for
~~~~~~~~~~~~~~~~~~~~~

This detector looks for **short-lived events** that appear suddenly and then relax exponentially back toward baseline.

A toy model is:

.. math::

   y(t) =
   \begin{cases}
   A\, e^{-(t-t_0)/\tau}, & t \ge t_0 \\
   0, & t < t_0,
   \end{cases}

where:

* :math:`A` is the event amplitude,
* :math:`t_0` is the start time,
* :math:`\tau` is the recovery timescale.

This configuration uses:

* decay time :math:`\tau = 5` days,
* search window :math:`= 20` days,
* minimum points :math:`= 5`,
* acceptance threshold :math:`\Delta \chi^2 \ge 16`,
* per-backend search,
* overlap suppression on,
* membership threshold :math:`|\mathrm{model}|/\sigma \ge 1`.

5.2 Why the search window is 20 days
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector searches a 20-day window for a 5-day decay. That is not arbitrary. Four e-folding times is enough to capture most of the recovery:

.. math::

   e^{-4} \approx 0.018.

So after 20 days, only about 1.8% of the original amplitude remains.

That means a 20-day window covers essentially the full event.

**Worked example**

If the event amplitude is :math:`A=12~\mu\mathrm{s}`:

* after 5 days: :math:`12e^{-1} \approx 4.4~\mu\mathrm{s}`,
* after 10 days: :math:`12e^{-2} \approx 1.6~\mu\mathrm{s}`,
* after 20 days: :math:`12e^{-4} \approx 0.22~\mu\mathrm{s}`.

So by 20 days, it has mostly died away.

5.3 What :math:`\Delta\chi^2 \ge 16` means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector compares two fits:

* a baseline model without the transient,
* a model with the transient.

The statistic is:

.. math::

   \Delta \chi^2 = \chi^2_{\text{baseline}} - \chi^2_{\text{event}}.

A larger value means the event model explains the data substantially better.

So requiring :math:`\Delta\chi^2 \ge 16` means “do not accept a transient unless it buys a real improvement in fit.”

That is a practical model-selection cutoff, not a magic number.

5.4 Why it runs per backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the search is backend-specific, the detector asks whether one system shows a transient-like pattern even if another does not.

That is useful because some apparent transients are instrumental. Running per backend lets the code distinguish:

* “something happened in one instrument”

from

* “something happened in the pulsar or line of sight.”

6. Exponential dip detector (``dip_cfg``)
-----------------------------------------

6.1 What it looks for
~~~~~~~~~~~~~~~~~~~~~

This detector looks for a **rapid drop followed by a slower recovery**. You can think of it as a negative transient with a more extended tail.

The rough time model is similar to the transient model but with negative amplitude and longer recovery:

.. math::

   y(t) \sim -A e^{-(t-t_0)/\tau_{\mathrm{rec}}}.

This configuration uses:

* recovery time = 30 days,
* search window = 150 days,
* minimum 6 points,
* minimum duration = 7 days,
* global scope,
* chromatic scaling :math:`\propto 1/f^\alpha` with :math:`\alpha \in [2,4]`.

6.2 Why the chromatic scaling matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many propagation effects in pulsar timing are frequency-dependent. The canonical cold-plasma dispersion delay scales like:

.. math::

   \Delta t \propto \nu^{-2},

so a chromatic model with :math:`1/f^\alpha` and :math:`\alpha` between 2 and 4 is probing whether the dip behaves more like a dispersive or scattering-related propagation effect than like an achromatic clock-like effect. Dispersion-like behavior is a central part of pulsar timing and DM analysis.

**Worked example**

If the fitted amplitude at 1400 MHz is :math:`1~\mu\mathrm{s}`:

* with :math:`\alpha=2`, the expected amplitude at 700 MHz is

  .. math::

     1 \times (1400/700)^2 = 4~\mu\mathrm{s};

* with :math:`\alpha=4`, it becomes

  .. math::

     1 \times (1400/700)^4 = 16~\mu\mathrm{s}.

So the same event can be much larger at lower observing frequency if it is chromatic.

6.3 Why global scope makes sense here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This detector is global, not per backend. That tells you the intended interpretation: it is looking for a real signal shared across the dataset, not an instrument-specific quirk.

7. Metadata merge (``merge_cfg``)
---------------------------------

This step merges libstempo TOAs with parsed timing-file metadata by **MJD and frequency**.

The tolerance is:

* about **10 seconds in time**,
* **1 MHz in frequency**.

libstempo is a Python wrapper around TEMPO2 and exposes TOAs, residuals, timing-model parameters, and the fit procedure. [3]_

**Worked example**

Suppose a TOA from libstempo says:

* MJD = 59000.123456,
* frequency = 1400.0 MHz.

And the metadata table has:

* MJD = 59000.12350,
* frequency = 1399.7 MHz.

If those differences are within the configured tolerances, the pipeline treats them as the same observation and merges them.

The point is simple: later detectors need the physics and geometry metadata attached to the correct TOAs.

8. Derived features (``feature_cfg``)
-------------------------------------

This stage adds explanatory variables that may help explain residual structure.

These include:

* orbital phase,
* solar elongation,
* elevation,
* airmass,
* parallactic angle,
* frequency bins.

These are sensible because timing residuals often vary with observing geometry or propagation path, not just with time. TEMPO2 itself emphasizes analysis of residuals against observational parameters such as parallactic angle or backend metadata. [1]_

**What each feature means**

**Orbital phase**
   Where the pulsar is in its binary orbit, usually mapped to a phase from 0 to 1.

**Solar elongation**
   Angular distance from the Sun on the sky. Small elongation means the line of sight passes closer to the solar wind.

**Elevation / airmass**
   How high the source was above the horizon. Lower elevation often means worse atmosphere and stronger telescope systematics.

**Parallactic angle**
   The geometric orientation of the source relative to the telescope feed. This can matter when polarization calibration is imperfect.

**Frequency bins**
   Coarse grouping of observing frequency, useful because many propagation effects are chromatic.

9. Feature-structure diagnostics and detrending (``struct_cfg``)
---------------------------------------------------------------

9.1 What this step does
~~~~~~~~~~~~~~~~~~~~~~~

This stage asks:

   “Do the residuals show systematic dependence on any of these explanatory features?”

It is set to ``mode="both"``, so it both:

* **tests** for structure, and
* **detrends** against it if present.

It uses:

* 16 bins per feature,
* minimum 3 points per bin,
* p-threshold = 0.02,
* orbital phase treated as circular,
* grouping by backend.

9.2 Why binning is useful
~~~~~~~~~~~~~~~~~~~~~~~~~

A simple way to detect structured dependence is to bin the data by a feature and ask whether the residual distribution changes from bin to bin.

**Worked example**

Suppose residuals are binned by elevation into 16 bins. If the low-elevation bins have a median residual near :math:`+2~\mu\mathrm{s}` while the high-elevation bins sit near zero, that suggests a geometry-linked systematic rather than a transient astrophysical event.

Detrending would then try to remove that smooth feature dependence before event detection.

9.3 Why orbital phase is circular
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Orbital phase wraps around, so phase 0.99 and phase 0.01 are close, not opposite. Treating orbital phase as an ordinary linear variable would be wrong. Circular treatment avoids artificial discontinuities.

10. Achromatic step detector (``step_cfg``)
-------------------------------------------

10.1 What it looks for
~~~~~~~~~~~~~~~~~~~~~~

This detector searches for a **step-like change** in the residuals that is **achromatic**, meaning it does not depend on radio frequency.

A simple model is:

.. math::

   y(t) = A\, H(t-t_0),

where :math:`H` is the Heaviside step function.

The settings are:

* enabled,
* scans both backend and global scope,
* minimum 10 points,
* :math:`\Delta\chi^2 \ge 12`,
* membership threshold :math:`1\sigma`,
* membership window up to 3650 days.

10.2 How to interpret an achromatic step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An achromatic step is the kind of thing you expect from:

* clock-like offsets,
* backend jumps,
* or genuinely broadband timing changes.

Since backend jumps are common in pulsar timing when mixing systems, it makes sense that this detector searches both backend-specific and global cases. [1]_

**Worked example**

Suppose that before MJD 59000 the residual mean is near 0, and after MJD 59000 it sits near :math:`+1.5~\mu\mathrm{s}` at all radio frequencies.

That is exactly the signature of an achromatic step.

11. Chromatic step detector (``dm_cfg``)
----------------------------------------

This is the frequency-dependent sibling of the achromatic step detector. It assumes the step scales approximately like:

.. math::

   y(t,\nu) \propto \nu^{-2},

which is the expected dispersion-law scaling for a DM-like change. Dispersion effects and DM analyses are central to pulsar timing because radio-wave propagation in plasma delays low-frequency signals more strongly than high-frequency ones.

The thresholds are the same as for ``step_cfg``, so the only real difference is the assumed chromatic law.

**Worked example**

If the step is :math:`1~\mu\mathrm{s}` at 1400 MHz, the same dispersive step would be roughly :math:`4~\mu\mathrm{s}` at 700 MHz.

So the detector is asking:

* “Did the mean level jump?”
* and also
* “Did it jump with the right frequency dependence for plasma propagation?”

12. Robust outlier detector (``robust_cfg``)
--------------------------------------------

This is the blunt-force generic detector:

* enabled,
* threshold :math:`= 5\sigma`,
* scans backend and global scopes.

A :math:`5\sigma` threshold is intentionally high. This detector is not for subtle structure; it is for cases where “that point is wildly inconsistent with the rest.”

**Worked example**

If a point has residual :math:`50~\mu\mathrm{s}` with formal uncertainty :math:`8~\mu\mathrm{s}`, its normalized significance is:

.. math::

   50/8 = 6.25\sigma.

That would exceed the threshold and be a candidate outlier.

This detector is useful because not every failure mode looks like a clean transient or step.

13. Detector preprocessing (``preproc_cfg``) and hard sigma gate (``gate_cfg``)
--------------------------------------------------------------------------------

**Preprocessing**

Preprocessing is disabled:

* ``detrend_features = []``
* ``use_preproc_for = []``

So the event detectors are not being run on an additional preprocessed version of the data.

If enabled, preprocessing would usually mean fitting and removing coarse group-level trends before searching for events.

**Hard sigma gate**

The hard gate is also disabled.

If it were enabled, the rule would be something like:

.. math::

   |r_i|/\sigma_i > 3

implies flag.

That is a crude but fast rule. Turning it off means the pipeline is choosing **model-based interpretation** over a simple threshold clip.

That is usually the right choice when you care about distinguishing astrophysical structure from junk.

14. Solar-event detector (``solar_cfg``)
----------------------------------------

14.1 What it looks for
~~~~~~~~~~~~~~~~~~~~~~

This detector searches for timing perturbations associated with the line of sight passing near the Sun.

That is astrophysically well motivated. Solar wind and solar-corona plasma affect pulsar timing, especially at small solar elongation and low radio frequency, and this is a known issue in high-precision timing. [4]_

The settings are:

* enabled,
* region: solar elongation :math:`\le 35^\circ`,
* fit exponential versus elongation,
* per-year fits allowed when enough data exist,
* otherwise global fallback,
* membership threshold :math:`0.8\sigma`,
* :math:`\alpha \in [0,4]`.

14.2 Why solar elongation matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solar elongation is the angle between the pulsar and the Sun on the sky.

* large elongation: the line of sight misses most of the dense solar wind,
* small elongation: the line of sight passes closer to the Sun, so plasma delays can increase.

Observational studies show that solar-wind timing effects become especially problematic near conjunction and at small elongation. [4]_

**Worked example**

If two observations are identical except that one is at :math:`60^\circ` elongation and the other at :math:`12^\circ`, the latter is much more likely to show extra chromatic delay from solar plasma.

That is why this detector only activates inside a geometric region.

14.3 Why per-year fits are allowed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solar wind is not exactly constant from year to year. Letting the detector fit each year separately, when enough data exist, is a way of acknowledging solar variability instead of forcing one global amplitude onto all years. Observational work on solar-wind timing corrections finds that time-variable solar-wind amplitudes fit better than a fixed-amplitude model, even if more sophisticated modeling is still needed. [4]_

15. Orbital-phase cut (``orbital_cfg``)
---------------------------------------

This is a legacy rule-based detector aimed at orbital-phase-dependent contamination.

It is enabled around phase 0.25 and uses a threshold based on **binned median significance**. It can mark points near that phase.

This is less a general model-fitting detector and more a “known troublesome orbital region” safeguard.

**Worked example**

Suppose a binary pulsar regularly shows suspicious excess residuals near orbital phase 0.25 because of eclipsing material, geometry, or calibration problems. Then this rule says: do not wait for a global fancy model; just inspect and mark points near that phase when they become significantly deviant.

This is pragmatic. Legacy cuts exist because some failure modes are repetitive and localized.

16. Eclipse detector (``eclipse_cfg``)
--------------------------------------

This detector is disabled.

That means the pipeline is **not** actively searching for eclipse-shaped events, even though the orbital-phase cut remains enabled.

So the configuration assumes eclipse-like behavior is either absent, handled elsewhere, or not worth fitting explicitly for this target.

17. Gaussian bump detector (``bump_cfg``)
-----------------------------------------

17.1 What it looks for
~~~~~~~~~~~~~~~~~~~~~~

This detector searches for broad hump-like deviations, modeled as Gaussian-shaped bumps:

.. math::

   y(t) = A \exp\!\left[-\frac{(t-t_0)^2}{2\sigma_t^2}\right].

The settings are:

* enabled,
* global scope,
* durations 20–2000 days,
* 10 trial scales,
* minimum 10 points,
* :math:`\Delta\chi^2 \ge 14`,
* no overlap suppression,
* membership threshold :math:`0.8\sigma`,
* :math:`\alpha \in [0,4]`.

17.2 Why scan multiple durations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Gaussian bump can be narrow or very broad. A 20-day event and a 1000-day event are qualitatively different, so the detector tries multiple widths.

**Worked example**

If the nominal duration is related to the Gaussian width, then:

* a 20-day candidate might look like a localized flare in the residuals,
* a 1000-day candidate might look like a broad, almost red-noise-like arch.

Scanning 10 trial scales is simply a discrete approximation to “search over possible event widths.”

17.3 Why no overlap suppression matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because overlap suppression is off, two partially overlapping bump candidates can both survive.

That is a deliberate choice, but it has a downside: broad smooth trends can sometimes be represented by several overlapping bumps. So this detector is flexible, but not automatically parsimonious.

18. Glitch detector (``glitch_cfg``)
------------------------------------

18.1 What a glitch is in pulsar timing terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A glitch is a sudden rotational irregularity, often idealized as a sharp change followed by relaxation. Glitches are a standard pulsar-timing phenomenon and are distinct from generic red timing noise. [5]_

The configuration is:

* enabled,
* minimum 14 points,
* :math:`\Delta\chi^2 \ge 14`,
* no overlap suppression,
* allows a peak+ramp model,
* 30-day peak timescale,
* noise-aware drop rule with ``noise_k=0.8``,
* rolling mean window 120 days,
* requires duration :math:`\ge 200` days.

18.2 What “peak + ramp” means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This suggests the model can include:

* a sharper local feature near the event epoch,
* plus a longer drift or relaxation afterward.

A toy schematic model might look like:

.. math::

   y(t) = A_{\text{peak}} e^{-(t-t_0)/\tau_{\text{peak}}} + B(t-t_0)H(t-t_0),

though the exact implementation may differ.

The point is not the precise formula; the point is that the detector can fit a **short-timescale response plus longer-term evolution**.

**Worked example**

Suppose a pulsar shows:

* an abrupt change near day 0,
* a strong short-timescale component over the next month,
* and then a slower trend over the next year.

A plain one-parameter step model would fit that badly. A peak+ramp model is much more plausible.

18.3 What the noise-aware drop rule is doing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration says the detector uses:

* ``noise_k = 0.8``,
* rolling mean window 120 days,
* minimum duration 200 days.

That is a safeguard against overfitting ordinary long-timescale red-noise wander as a glitch.

The basic logic is:

* estimate what the noise background looks like over a rolling window,
* compare the candidate event against that background,
* reject weak candidates that do not rise far enough above it.

That is what “noise-aware” should mean in practice.

19. Why some detectors are backend-specific and others are global
-----------------------------------------------------------------

This is worth spelling out because it is one of the most intelligent parts of the configuration.

**Backend-specific searches**

Used when an effect could easily be instrumental:

* transient detector,
* achromatic step detector,
* robust outlier detector.

These ask: “Does one system misbehave on its own?”

**Global searches**

Used when an effect is intended to be astrophysical or line-of-sight dependent:

* dip detector,
* solar detector,
* bump detector.

These ask: “Does the pattern appear across the combined dataset in a physically meaningful way?”

That split is not decorative. It encodes prior beliefs about which kinds of phenomena should be shared across systems.

20. How the significance thresholds work in practice
----------------------------------------------------

Several detectors use:

* minimum point counts,
* :math:`\Delta\chi^2` thresholds,
* membership thresholds such as :math:`0.8\sigma` or :math:`1\sigma`.

These are doing different jobs.

**Minimum point count**

Prevents single-point or two-point coincidences from masquerading as events.

**:math:`\Delta\chi^2` threshold**

Ensures the event model improves the fit enough to be worth keeping.

**Membership threshold**

Determines which TOAs count as part of the event once a candidate is found.

**Worked example**

Imagine a candidate step model predicts :math:`+2~\mu\mathrm{s}` for a TOA whose uncertainty is :math:`1.5~\mu\mathrm{s}`. Then:

.. math::

   |model|/\sigma = 2/1.5 \approx 1.33.

If the membership threshold is :math:`1\sigma`, that point is a member.

If another point has model amplitude :math:`0.6~\mu\mathrm{s}` and uncertainty :math:`1.2~\mu\mathrm{s}`, then:

.. math::

   0.6/1.2 = 0.5,

so it would not count as a member.

So “membership” is not the same thing as “being in the time window.” It means the fitted event predicts a large enough effect at that TOA.

21. The semantic rule that changes everything
---------------------------------------------

**Rule**

If a point is identified as an **event member**, then it is treated as **not a generic bad point** for the purposes of ``bad_point``.

But the summary flag ``outlier_any`` is still true if either:

* the point is a bad point, or
* the point is an event member.

So:

.. math::

   \texttt{outlier_any} = \texttt{bad_point} \lor \texttt{event_member}.

**Why this is a good policy**

Without this rule, the pipeline would penalize real events for being unusual.

A chromatic dip caused by propagation, or a step caused by a real systematic change, should be labeled “special,” not “garbage.”

So the configuration draws a three-way distinction:

1. **normal point** — nothing special,
2. **bad point** — probably junk,
3. **event member** — unusual but potentially meaningful.

That is a much better taxonomy than “everything weird is trash.”

22. A fully plain-English summary of the whole config
-----------------------------------------------------

This pipeline starts from a pulsar timing model and TOAs, merges in metadata, adds physically meaningful features, and then runs several detectors, each representing a different hypothesis about why residuals look strange.

The bad-point detector uses a short-memory correlated-noise model rather than pretending nearby measurements are independent. It controls false alarms using a 2% false-discovery-rate rule, and it can mark multiple bad TOAs on the same day.

Other detectors then look for specific classes of coherent structure:

* short transients with exponential recovery,
* dips with slower recovery and chromatic scaling,
* achromatic or DM-like steps,
* solar-wind-related events near conjunction,
* broad Gaussian bumps,
* and glitch-like behavior.

The structure-diagnostics stage checks whether residual patterns are better explained by observing geometry or orbital phase than by real events.

Most importantly, if a point belongs to a coherent event model, the pipeline does not treat it as an ordinary bad point. It still counts as “unusual,” but it is not treated as “junk.”

That is the core philosophy of the configuration:
**be strict about bad data, but do not throw away structured signals just because they are inconvenient.**

23. Suggested references for readers
------------------------------------

These are useful starting points for the ideas used here:

**Pulsar timing and timing residuals**

* Hobbs et al., *TEMPO2, a new pulsar-timing package – I. An overview* — explains TOAs, timing residuals, system jumps, and analysis workflow. [1]_
* Hobbs, Lyne, Kramer, *Pulsar Timing Noise* — concise overview of timing residuals, noise, and glitch-related structure. [5]_
* libstempo documentation — practical wrapper around TEMPO2 for loading ``par``/``tim`` files and accessing TOAs and residuals. [3]_

**False discovery rate**

* Benjamini & Hochberg (1995), *Controlling the False Discovery Rate* — the classic BH multiple-testing paper. [2]_

**Dispersion / chromatic effects**

* Kulkarni, *Dispersion measure: Confusion, Constants & Clarity* — useful discussion of dispersion observables and :math:`\nu^{-2}`-type delay behavior.
* Iraci et al. (2024), on DM time-series methods in pulsar timing — modern comparison of DM-variation approaches. [6]_

**Solar-wind timing contamination**

* Tiburzi et al. (2021), *The impact of solar wind variability on pulsar timing* — why low elongation and solar conjunction matter. [4]_

24. A shorter “TL;DR” version
-----------------------------

   This pipeline is a pulsar timing quality-control and event-detection system. It starts from the timing model and the observed pulse arrival times, computes residuals, merges in metadata, and adds physically useful features such as orbital phase and solar elongation.

   The first major detector looks for bad points, but it does not use a naive independent-noise model. Instead, it assumes nearby observations can be correlated over about 30 minutes using an Ornstein–Uhlenbeck process, which is a mean-reverting correlated-noise model. It then controls false alarms using a 2% false discovery rate, so the final list of bad points stays strict even though many tests are being performed.

   After that, the pipeline searches for specific structured phenomena: short transients, longer chromatic dips, achromatic or DM-like steps, solar-wind events near the Sun, broad bumps, and glitch-like signatures. Some searches are run per backend when the effect might be instrumental; others are global when the effect is intended to be astrophysical.

   A separate structure-analysis stage checks whether trends are better explained by things such as orbital phase, elevation, or parallactic angle. That helps avoid mistaking observing geometry for real events.

   The key policy choice is that once a point is identified as belonging to a coherent event, it is no longer treated as just a random bad point. It is still marked as unusual, but it is interpreted as structured rather than junk.

References
----------

.. [1] https://www.atnf.csiro.au/research/pulsar/psrcat/mnras0369-0655.pdf
.. [2] https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x
.. [3] https://vallis.org/libstempo/
.. [4] https://openaccess.inaf.it/bitstreams/ccde2595-3f3f-491b-80a9-2fca90091e1d/download
.. [5] https://www.raa-journal.org/issues/all/2006/v6ns2/202203/P020220325510334550552.pdf
.. [6] https://www.boa.unimib.it/retrieve/d5999fa6-3891-4cc6-81c0-5fb1d87a55b1/Iraci-2024-Astronomy%20Astrophys-VoR.pdf
