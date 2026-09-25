# Contributing

Thank you for contributing to Diffusers Engine Q21.

## Development rules

1. Create changes from the latest default branch.
2. Keep model weights, virtual environments, generated outputs, secrets, and
   private reference images out of Git.
3. Use the existing startup profiles instead of adding undocumented environment
   combinations.
4. Update README and `docs/performance-low-vram.md` when behavior, VRAM, or
   benchmark methodology changes.
5. Run the checks below before opening a pull request.

```bash
bash -n start.sh start_backend.sh start_frontend.sh scripts/start_options.sh
.venv/bin/python -m compileall -q backend scripts
node --check frontend/app.js
.venv/bin/python -m pip check
```

GPU changes should include a benchmark Markdown/JSON pair produced by
`scripts/benchmark_gpu.py`. Record the GPU, driver, software versions, steps,
resolution, reference count, speed, and peak VRAM. Do not use 2-step results as
practical speed comparisons.

## Licensing

Unless explicitly stated otherwise, contributions intentionally submitted to
this repository are provided under the Apache License 2.0, consistent with
Section 5 of that license. Do not submit code, weights, images, or documentation
that you do not have the right to redistribute.

Model weights and adapters are not covered by the project's Apache License.
See `THIRD_PARTY_NOTICES.md` before changing model download or redistribution
behavior.
