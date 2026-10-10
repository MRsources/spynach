# spinpy documentation

| guide | notebook | covers |
|---|---|---|
| [User guide](user-guide.md) | [user-guide.ipynb](notebooks/user-guide.ipynb) | liquid-state NMR in Hilbert space (`zeeman-hilb`): setup, the simulation pipeline, pulse sequences, tolerances, parity with Spinach |
| [Coming from Spinach](coming-from-spinach.md) | [coming-from-spinach.ipynb](notebooks/coming-from-spinach.ipynb) | what carries over from Spinach and what differs: spin systems, basis, operators, Hamiltonians, parameters, calculation modalities, processing |

Open the notebooks with `OPENBLAS_NUM_THREADS=4 jupyter lab docs/notebooks/`
after `pip install -e ".[docs]"`.

## Conventions for contributors

**Every user-facing change ships with user documentation in the same
commit.** A newly ported function, formalism, context, pulse sequence,
option, or behaviour change is not done until the guides describe it.

1. **Markdown is the source.** Guides are `docs/<name>.md`. Every
   ```` ```python ```` block is executed by `tests/docs/test_guides.py`, in
   order and in one namespace per guide, so examples must run as written
   and should assert what they claim.
2. **Notebooks are generated.** `python docs/build_notebooks.py` writes
   `docs/notebooks/<name>.ipynb` for every guide, without outputs. Never edit
   a notebook by hand; the tests fail if one is stale or has no guide.
3. **Plots** go in their own blocks, use matplotlib, and define no names that
   later blocks need: without matplotlib installed, the tests skip them.
4. **Keep examples fast** (a few seconds per guide). If a faithful
   translation of a Spinach example is slow, reduce its size and say what
   was changed and how to restore it.
5. **Compare with Spinach.** For every ported feature, update
   [Coming from Spinach](coming-from-spinach.md): what behaves the same,
   what differs (call syntax, defaults, return types, units, numerics), and
   what is still not ported. Move items out of the "not ported" lists of
   both guides and the README status table as they are ported.
6. **Split guides by theme as they grow.** When a guide passes about 500
   lines, or a new formalism, context, or experiment family arrives, give it
   its own guide (for example `liouville-space.md`, `relaxation.md`,
   `solid-state.md`) rather than appending, and add it to the table above.
   Its notebook follows automatically.
7. Use 0-based spin numbers, Spinach's field names and units, and the
   verified tolerances; link to sections instead of repeating them.
