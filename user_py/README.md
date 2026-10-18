# User-contributed Python code

Notebooks and scripts contributed by Spynach users, kept as received
with their inputs. They are the Python counterpart of `user_ml/`
(contributed Spinach scripts): nothing here is a translation of a
Spinach example, so there is no comparison case unless the author
provides a MATLAB equivalent. They are not run by the test suite.

| file | author | what it calculates | inputs |
|---|---|---|---|
| `na23_tqf_buildup.ipynb` | Alexej Jerschow, 2026 | triple-quantum build-up of a single 23Na spin under quadrupolar Redfield relaxation towards thermal equilibrium (C_q = 200 kHz, tau_c = 1 ns, 14.1 T, 298 K): from the thermal state turned transverse by a 90-degree pulse, the rank-3 single-quantum tensor, the triple-quantum coherence after a second pulse and the longitudinal recovery, 0 to 10 ms in 200 steps, against the biexponential form of Jaccard, Wimperis and Bodenhausen; tensor states by their `Tl,m` labels | none |

To run a notebook, install the `docs` extra (`pip install -e ".[docs]"`)
and open it with `jupyter lab user_py/`; the notebooks are saved with
their outputs, so the plots show without running them. In a container
whose CPU quota is below its visible core count, set
`OPENBLAS_NUM_THREADS` to the quota first (see the setup section of
[`docs/hilbert-space.md`](../docs/hilbert-space.md#setup)).

## Contributing

Add the file with a header or first cell that names the author and the
year, add a row to the table, keep any input files next to it, and keep
the notebook fast (seconds) so it can be re-run when Spynach changes.
Plot with `spynach.plotting` where it fits.

## Licence

Contributed code and input files remain the copyright of their
authors, named in the table above, and are redistributed with their
agreement under the repository's MIT licence (`LICENSE` at the root).
Keep the author's name in the file and in this table.
