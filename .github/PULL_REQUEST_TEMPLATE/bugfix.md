---
name: Bugfix
about: Fix a bug with tests and impact notes
title: "fix: "
labels: bug
assignees: ""
---

## Problem

What breaks today? Include concrete symptoms and scope.

## Root cause

What caused the issue?

## Fix

What changed to fix it?

## Risk assessment

- Any behavior changes?
- Any compatibility concerns?

## Validation

- [ ] Added/updated regression tests
- [ ] `python -m ruff check .`
- [ ] `python -m black --check .`
- [ ] `python -m pytest`

## Example

```
Problem: exp_dip points were still eligible for glitch assignment.
Root cause: glitch scanner did not exclude exp_dip_member rows.
Fix: add exclusion mask and regression test.
Risk: low, only narrows glitch candidates.
```
