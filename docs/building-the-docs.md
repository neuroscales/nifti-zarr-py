# Building the docs

This site is built with [Zensical](https://zensical.org) from the
`mkdocs.yml` file at the root of the
[nifti-zarr-py](https://github.com/neuroscales/nifti-zarr-py) repository.

## Preview locally

The specification is included from the `nifti-zarr-schema` submodule, so
check it out first:

```bash
git submodule update --init
```

Install the documentation dependencies (requires pip >= 25.1):

```bash
pip install --group docs
```

Start a live preview at <http://localhost:8000>:

```bash
zensical serve
```

To build the static site into `site/` instead, failing on any warning:

```bash
zensical build --strict
```

## Where the content comes from

Most pages are generated from existing files, so edit those rather than the
files in `docs/`:

| Page | Source |
| ---- | ------ |
| Home | `README.md` |
| Specification | `nifti-zarr-schema/README.md` ([neuroscales/nifti-zarr](https://github.com/neuroscales/nifti-zarr)) |
| JSON schema | `nifti-zarr-schema/nifti-zarr-schema-1.0.rc1.json` |
| API reference | Docstrings in the `niizarr` package |

## Publishing

The `Docs` GitHub Actions workflow builds the site on every pull request and
publishes it to <https://nifti-zarr.lincbrain.org> on every push to `main`.
