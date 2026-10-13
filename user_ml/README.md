# User-contributed Spinach scripts

Scripts contributed by users, kept as received, with their inputs. Each
has a line-by-line spinpy translation in `examples/user_ml/` and a
comparison case in `comparison/cases/` (see `comparison/README.md` for
the timed, profiled Spinach-vs-spinpy protocol and its report).

| script | author | translation | comparison case | inputs |
|---|---|---|---|---|
| `ZULF_PF6_1D.m` | Florin Teleanu, 2024 | `examples/user_ml/zulf_pf6_1d.py` | `zulf_pf6_1d` | `rho_adiabatic_drop.csv` (state after the adiabatic drop, MATLAB `csvwrite` format) |

Notes on `ZULF_PF6_1D.m`: it targets a pre-2.9 Spinach. On 2.13 the
comparison case applies two fixes, mirrored in the translation:
`rPF*{...}` (a number times a cell array) is written out, and the legacy
`apodization(fid,'exp-1d',1)` becomes `apodisation(spin_system,fid,{{'exp',1}})`.
The `csvread` path is replaced by the file in this directory. The
commented-out saturation-transfer scan at the end of the script
(`shaped_pulse_af`, `zerofield`) is not translated.

## Licence

Contributed scripts and their input files remain the copyright of their
authors, named in the table above, and are redistributed with their
agreement under the repository's MIT licence (`LICENSE` at the root).
Keep the author's name in the script header and in this table.
