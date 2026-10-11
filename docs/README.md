# spinpy documentation

| guide | notebook | covers |
|---|---|---|
| [Hilbert space](hilbert-space.md) | [hilbert-space.ipynb](notebooks/hilbert-space.ipynb) | liquid-state NMR in Hilbert space (`zeeman-hilb`): setup, the simulation pipeline, pulse sequences, tolerances, parity with Spinach |
| [Liouville space](liouville-space.md) | [liouville-space.ipynb](notebooks/liouville-space.ipynb) | the spherical-tensor formalism (`sphten-liouv`): basis descriptor, state-space restriction (`IK-0/1/2`), filters, chemical subsystems |
| [Solid state](solid-state.md) | [solid-state.ipynb](notebooks/solid-state.ipynb) | anisotropic interactions: tensors from eigenvalues and Euler angles, dipolar couplings from coordinates, quadrupolar couplings, the `Q` components and `orientation`, `crystal` and `powder` contexts, grids, assumption sets |
| [Relaxation](relaxation.md) | [relaxation.ipynb](notebooks/relaxation.ipynb) | Redfield theory for rotational diffusion, T1/T2 and damping models, retention policies, thermal equilibrium and thermalisation, transient NOE |
| [Magic angle spinning](magic-angle-spinning.md) | [magic-angle-spinning.ipynb](notebooks/magic-angle-spinning.ipynb) | the Fokker-Planck treatment of sample rotation: rotor phase grid and spectral derivative, the `singlerot` context, sidebands, rotor echoes |
| [Pulse sequences](pulse-sequences.md) | [pulse-sequences.ipynb](notebooks/pulse-sequences.ipynb) | writing sequences, `apodisation` and `plot_2d`, decoupling and homospoil, COSY, NOESY, HSQC with isotopomer dilution |
| [Coming from Spinach](coming-from-spinach.md) | [coming-from-spinach.ipynb](notebooks/coming-from-spinach.ipynb) | what carries over from Spinach and what differs: spin systems, basis, operators, Hamiltonians, parameters, calculation modalities, processing |

The notebooks are saved with their outputs, plots included. Open them
with `OPENBLAS_NUM_THREADS=4 jupyter lab docs/notebooks/` after
`pip install -e ".[docs]"`, or in VS Code with the Python and Jupyter
extensions; the plots are interactive in both. (GitHub's notebook viewer
does not show Plotly figures.)

To experiment, copy a notebook into `scratch/` at the repository root
(ignored by git) and work on the copy; it uses the same `spinpy` code. The
notebooks in `docs/notebooks/` are rebuilt from the guides, so edits to them
are overwritten; if you run one there, `git restore docs/notebooks/` discards
the changed outputs.

## Conventions for contributors

**Every user-facing change ships with user documentation in the same
commit.** A newly ported function, formalism, context, pulse sequence,
option, or behaviour change is not done until the guides describe it.

1. **Markdown is the source.** Guides are `docs/<name>.md`. Every
   ```` ```python ```` block is executed by `tests/docs/test_guides.py`, in
   order and in one namespace per guide, so examples must run as written
   and should assert what they claim.
2. **Notebooks are generated.** `python docs/build_notebooks.py` writes
   `docs/notebooks/<name>.ipynb` for every guide, runs it in a fresh kernel,
   and saves it with its outputs. Never edit a notebook by hand; the tests
   fail if one is stale, was saved without outputs, or has no guide.
3. **Plots** go in their own blocks, use `spinpy.plotting` (or Plotly
   directly), end with the figure as the last expression so the notebook
   shows it, and define no names that later blocks need: without Plotly
   installed, the tests skip them. Show the results a section computes:
   spectra, FIDs, and comparisons between them.
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
