## Summary

Describe what changed and why.

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Docs update
- [ ] Refactor
- [ ] Tests

## Validation

- [ ] `python -m ruff check .`
- [ ] `python -m black --check .`
- [ ] `python -m pytest`
- [ ] `make -C docs html` (if docs/API changed)

## Checklist

- [ ] Tests added/updated for changed behavior
- [ ] Docs updated for user-facing changes
- [ ] Backward compatibility considered
- [ ] No unrelated files changed

## Example (good PR description)

```
This PR fixes false glitch labeling on exp-dip recovery tails.

Changes:
- exclude exp_dip members from glitch candidate pool
- add regression test for J1713-like dip pattern
- update docs with event precedence rules

Validation:
- ruff, black, pytest pass locally
```
