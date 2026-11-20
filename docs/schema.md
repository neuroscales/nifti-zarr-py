# JSON schema

The JSON variant of the NIfTI header, stored in the attributes of the
`nifti` array (see [section 2.4 of the specification](specification.md#24-nifti-header)),
is described by the JSON schema below. Its canonical location is
<https://raw.githubusercontent.com/neuroscales/nifti-zarr/main/nifti-zarr-schema-1.0.rc1.json>.

To validate the JSON header of a NIfTI-Zarr file:

```python
import json
import urllib.request

import jsonschema
import zarr

url = "https://raw.githubusercontent.com/neuroscales/nifti-zarr/main/nifti-zarr-schema-1.0.rc1.json"
with urllib.request.urlopen(url) as f:
    schema = json.load(f)

header = dict(zarr.open_group("path/to/mri.nii.zarr", mode="r")["nifti"].attrs)
jsonschema.validate(header, schema)
```

## Schema

```json title="nifti-zarr-schema-1.0.rc1.json"
--8<-- "nifti-zarr-schema/nifti-zarr-schema-1.0.rc1.json"
```
