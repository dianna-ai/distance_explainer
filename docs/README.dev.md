# `distance_explainer` developer documentation

If you're looking for user documentation, start from the [documentation home](index.md).

## Development install

```shell
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip setuptools
python3 -m pip install --no-cache-dir --editable ".[dev]"
```

## Running tests

```shell
pytest -v
```

Or using `tox` (install separately with `pip install tox`):

```shell
tox
```

### Test coverage

```shell
coverage run && coverage report
```

## Running linters

```shell
ruff check .
```

To run linters automatically on commit, enable the git hook:

```shell
git config --local core.hooksPath .githooks
```

## Versioning

Bump the version across all files with [bump-my-version](https://github.com/callowayproject/bump-my-version):

```shell
bump-my-version bump major
bump-my-version bump minor
bump-my-version bump patch
```

## Making a release

Releases are fully automated via [GitHub Actions](https://github.com/dianna-ai/distance_explainer/blob/main/.github/workflows/release.yml). To make a release:

1. Create a [GitHub Release](https://github.com/dianna-ai/distance_explainer/releases/new) with a new tag.
2. The workflow automatically builds and uploads the package to PyPI.

For a test upload to TestPyPI, manually trigger the [Build and upload to PyPI](https://github.com/dianna-ai/distance_explainer/actions/workflows/release.yml) workflow via the GitHub Actions UI with `workflow_dispatch`.

## Generating the API docs

```shell
cd docs
make html
```

Documentation will be in `docs/_build/html`. If you don't have `make`:

```shell
sphinx-build -b html docs docs/_build/html
```
