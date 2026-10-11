# spinpy: notes for agents

Read `HANDOFF.md` for goals and design decisions, `README.md` for status,
`porting/ROADMAP.md` for the current step, and `porting/design-notes.md`
for the Liouville-space kernel design (literature and Spinach source review).

## User documentation is part of every change

- Every user-facing change (new port, option, behaviour change) includes
  user documentation in the same commit. Follow the conventions in
  `docs/README.md`.
- Guides are `docs/*.md`; their `python` blocks are executed by
  `tests/docs/`. Run `python docs/build_notebooks.py` after editing a guide:
  notebooks in `docs/notebooks/` are generated and saved with their outputs
  (plots included), never hand-edited. Plot with `spinpy.plotting`.
- For each ported feature, update `docs/coming-from-spinach.md` with what is
  the same as in Spinach, what differs, and what is still not ported.
- Split a guide by theme when it passes ~500 lines or a new formalism,
  context, or experiment family arrives; add it to the table in
  `docs/README.md`.

## Testing

```
OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q
```

The `OPENBLAS_NUM_THREADS` setting matters in this container (4-CPU quota,
12 visible cores).
