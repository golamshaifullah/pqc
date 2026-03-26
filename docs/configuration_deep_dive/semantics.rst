Event/Outlier Semantics
=======================

Why semantics matter
--------------------

A point can be unusual because it is garbage, or because it belongs to a real,
coherent event. PQC keeps these concepts separate.

Core flags
----------

- ``bad_point``: generic outlier/bad-measurement status.
- ``event_member``: membership in accepted event models.
- ``outlier_any``: compatibility aggregate used by some consumers.

Current policy
--------------

If a TOA is part of an accepted event model, it is treated as a non-outlier in
``bad_point``. This prevents coherent event structure from being mislabeled as
random bad data.

Compatibility aggregate
-----------------------

``outlier_any`` is retained as:

.. math::

   \texttt{outlier\_any} = \texttt{bad\_point} \lor \texttt{event\_member}

So event members can still appear in ``outlier_any`` even when they are not
``bad_point``.

Three-way interpretation
------------------------

1. **Normal**: not bad, not event.
2. **Bad**: bad point with no coherent event support.
3. **Event member**: unusual but structured and interpretable.

This taxonomy is the intended behavior for downstream QC review.
