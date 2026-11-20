# Building the docs

This site is built with [Zensical](https://zensical.org), which reads the
`mkdocs.yml` file at the root of the
[nifti-zarr-py](https://github.com/neuroscales/nifti-zarr-py) repository
natively.

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

The Specification and JSON schema pages are rendered from the
[neuroscales/nifti-zarr](https://github.com/neuroscales/nifti-zarr)
repository, which is included here as the `nifti-zarr-schema` submodule.

The READMEs and the schema are included with the
[`pymdownx.snippets`](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/)
extension, so they stay the single source of truth.

## Publishing

The `Docs` GitHub Actions workflow (`.github/workflows/docs.yml`) builds the
site on every pull request, and publishes it to GitHub Pages at
<https://nifti-zarr.lincbrain.org> on every push to `main`.

Only the upstream repository,
[neuroscales/nifti-zarr-py](https://github.com/neuroscales/nifti-zarr-py),
publishes the site. Forks and pull requests only build it.
