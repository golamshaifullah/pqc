# PQC Release Checklist

## 1) Clean and verify

```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

## 2) Build docs

```bash
cd docs
make html
cd ..
```

## 3) Build distributions

```bash
python -m build
python -m twine check dist/*
```

## 4) Test install from wheel/sdist

```bash
python -m pip install --force-reinstall dist/*.whl
python -c "import pqc; print('ok')"
```

For pipeline execution support:

```bash
python -m pip install --force-reinstall "dist/*.whl[libstempo]"
```

## 5) Publish

TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

PyPI:

```bash
python -m twine upload dist/*
```
