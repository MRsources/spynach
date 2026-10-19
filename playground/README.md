# Playground

A place to experiment with notebooks and scripts. Everything in this folder
except this README is ignored by git, so nothing here is committed by
accident and the test suite never looks at it.

- Copy a guide notebook here to play with it: `cp docs/notebooks/ctps.ipynb playground/`.
  The copies in `docs/notebooks/` are regenerated from the guides, so edits
  there are overwritten; edits here are yours to keep.
- Select the project environment as the kernel (`.venv/bin/python` in the
  container, `.venv-mac/bin/python` on the Mac). `import spynach` and
  `import ctps` work from here; in a container whose CPU quota is below its
  visible core count, set `OPENBLAS_NUM_THREADS` to the quota first.
- When an experiment turns into something worth keeping, move it to
  `user_py/` with your name in its first cell, or into a guide.
