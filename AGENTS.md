# Repository instructions

Read [CONTRIBUTING.md](CONTRIBUTING.md) before editing.

- Make one small, reviewable change at a time. Do not generate large batches of code, experiments or documents in one step.
- Inspect status and source first; review the staged diff and run relevant checks before each commit. Leave unrelated untracked research files alone.
- Publish only redacted research metadata. Keep credentials, private server paths, logs, datasets and checkpoints out of commits and public reports.
- Scan for privacy leaks before publishing. Report findings by file and type; never echo a secret value.
- Run training and framework validation on the remote Agent. Preserve running jobs and immutable source snapshots during source privacy cleanup; local checks stay static or use the standard library.
- Commit inertial research here; commit remote terminal transport changes to the separate `app-remote-terminal` repository.
