# AutoDL v1 operations

These tools manage the independent dev/v1 installation at
`/root/autodl-tmp/other-inertial-benchmark-v1`. They use the existing `NIO` API,
v1 converters and v1 configuration keys. They do not arm a GPU campaign.

Install the optional client dependencies with `pip install -e '.[ops]'`. Copy
[ssh-config.example.json](ssh-config.example.json) to `.local/autodl-v1.json`,
fill in the private SSH connection fields, and run `chmod 600 .local/autodl-v1.json`.
This machine configuration is ignored by Git and excluded from deployment.
Keep `.local/autodl.json` for historical server assets if it already exists.

```bash
python tools/ops_v1/remote.py status
python tools/ops_v1/remote.py sync
python tools/ops_v1/remote.py exec 'cd /root/autodl-tmp/other-inertial-benchmark-v1 && .venv/bin/ipb info'
```

Deployment uses admitted tracked and nonignored source files, excludes symlinks,
datasets, runs, external assets and private configuration, and records exact
SHA-256 hashes. A source archive is staged before the shared CPU/GPU locks are
acquired. If a lock is busy, exit code 75 means the staged snapshot has not been
installed; retry after the active task finishes. A full deployment removes only
previously managed source files omitted by the new snapshot, after checking
their previous hashes. Historical archives, data and results are preserved.

On the server, use the shared CPU lock for acceptance and conversion tasks:

```bash
cd /root/autodl-tmp/other-inertial-benchmark-v1
/root/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --cache-dir .local/pip-cache -e '.[train,test,ops]'
.venv/bin/python -m pip install --cache-dir .local/pip-cache --ignore-installed --no-deps ruff==0.13.2
source .venv/bin/activate
export IPB_DATASETS="$PWD/datasets/ipb-v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c 'import inertial_benchmark; print(inertial_benchmark.__file__)'
ipb info
flock /root/autodl-tmp/inertial-benchmark/.local/campaigns/cpu-resource.lock python tools/ops_v1/server_acceptance.py --stage cpu-acceptance
flock /root/autodl-tmp/inertial-benchmark/.local/campaigns/cpu-resource.lock python tools/ops_v1/run_checks.py --stage cpu-file-acceptance --pytest-isolation=file
```

The independent virtual environment shares the installed Torch wheels and
selects the new editable distribution before historical system packages. Keep
the old environment installed. `ipb` and package imports should work without
`PYTHONPATH`; the printed package path must point to this v1 directory.
`run_checks.py` records pytest, Ruff, compileall, package location, `ipb info`,
resolved config and benchmark dry-run with the exact deployment manifest hash.
Use a new `--stage` name for each attempt; the tool refuses to overwrite existing
evidence. The Ruff install gives the independent environment its own CLI while
retaining the historical environment.

`server_acceptance.py` reads RIDI and RoNIN raw recordings from the historical
server directory, converts three recordings per dataset with the existing v1
converters, and retains their complete converted outputs. It then makes separate
800-sample excerpts from distinct train, val and test recordings. The full
`ronin_resnet18` receives four training windows, batch two and one epoch, producing
two optimizer updates per dataset. Checkpoint reload, val, prediction, NPZ/PNG
outputs and exact prediction equality are checked. These are engineering
acceptance results; the short excerpts are unsuitable for research ranking,
convergence or generalization claims. Train/eval mismatch diagnostics remain
visible, and export is not part of this acceptance.

`cpu_pytest.py` restores one Torch thread after the repository test conftest is
loaded. Test fixtures remain unchanged. Tests requiring unavailable full dataset
manifests skip explicitly; a passed synthetic test is not a public-data run.
On the 2 GiB CPU instance, `--pytest-isolation=file` runs each original test module
in a fresh process. It collects nodeids first and rejects missing, duplicated or
unfinished nodes. Per-module JUnit, logs and atomic node reports survive an
interrupted test process. The default `session` mode runs the usual whole suite.
Both modes set `MALLOC_ARENA_MAX=2` and record allocator, BLAS and Torch thread
settings; this changes the resource environment while preserving test parameters.
Repeat `--pytest-path tests/test_views.py` for an explicit affected-module scope;
the receipt distinguishes this scope from a full run. Explicit `IPB_RAW_*`
environment variables select available real raw sources and are recorded in the
receipt. The tool does not create source views automatically.

Use `datasets/ipb-v1/<dataset>/` for full v1 conversions and the `IPB_DATASETS`
variable to select this root. `_smoke` and `_cpu_excerpt` directories are private
acceptance scopes. Source raw files and historical canonical files are never
rewritten by these tools. The GPU profile is only a starting configuration:
keep `amp=false`, measure batch capacity and throughput per model after CUDA
becomes available, cover one seed first, and use at most three seeds for core
research comparisons. Seed 42 in these engineering profiles does not replace
the research matrix's initial seed 0.
