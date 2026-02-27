---
name: Feature
about: Add a new feature with docs/tests
title: "feat: "
labels: enhancement
assignees: ""
---

## Feature summary

What is added and for whom?

## Design

- Key API/config changes
- Tradeoffs and alternatives considered

## Implementation

Main files/modules touched.

## Validation

- [ ] Unit/integration tests added
- [ ] Docs updated (README/docs/api docstrings)
- [ ] `python -m ruff check .`
- [ ] `python -m black --check .`
- [ ] `python -m pytest`
- [ ] `make -C docs html` (if docs/API changed)

## Example

```
Adds a global solar-event detector replacing the old solar cut.
Config: SolarCutConfig now controls event detection + handling.
Behavior: solar_event_member rows are non-outliers in bad_point.
```
