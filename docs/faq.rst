FAQ
===

Why do I only see API reference pages?
--------------------------------------

Make sure you are building the docs from the updated Sphinx sources. The
index page should include narrative sections such as Overview, Installation,
Quickstart, Concepts, and Examples. If the TOC looks sparse, run a clean build
(``make clean && make html``).

How do I plot event membership?
-------------------------------

Use :func:`pqc.utils.diagnostics.event_membership_mask` to build a boolean
mask for plotting. The default uses informative membership for step and
DM-step events; set ``use_applicable=True`` to include applicable points.

What does FDR mean here?
------------------------

FDR is the expected fraction of false positives among the declared discoveries.
PQC uses the Benjamini-Hochberg procedure to control FDR when flagging bad
measurements [BH1995]_.

References
----------

.. [BH1995] Benjamini, Y., & Hochberg, Y. (1995).
   "Controlling the false discovery rate: a practical and powerful approach to multiple testing."
   *Journal of the Royal Statistical Society Series B*, 57(1), 289-300.
   citeturn1search0
