# tools — one-off utilities

Scripts that are run by hand, occasionally, to produce something that then lives in the repo. Nothing
in the normal path imports them. If a thing runs on every kernel, it belongs in
[`app/`](../app/README.md) instead.

| file | role |
|---|---|
| `extract_model.py` | pull a datapath model out of an upstream checkout into `app/mxmesh/`, so the repo does not depend on that checkout at run time |

## Using it

```bash
.venv/bin/python tools/extract_model.py --help
```

What is extracted is then pinned by `tests/selftest_extracted.py`, which fails if `app/mxmesh/` has
drifted from the source it came from. Re-extract, don't hand-edit.

See [`../README.md`](../README.md) for install and the run command.
