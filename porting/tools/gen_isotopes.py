"""Generate src/respinach/_isotope_data.py from Spinach's kernel/spin.m.

Usage: python porting/tools/gen_isotopes.py /spinach

Parses the single-name ``case`` entries of the isotope switch (multiplicity
and magnetogyric ratio) and the list of isotopes with no literature data.
The electron-like ('E#') and bosonic ('C#', 'V#', 'T#') name patterns are
handled in code, not in the table.
"""

import re
import subprocess
import sys
from pathlib import Path

ENTRY = re.compile(
    r"case\s+'([^']+)'[^\n]*\n\s*multiplicity=\s*([0-9]+)\s*;\s*\n\s*gamma=\s*([-+0-9.eE]+)\s*;",
)
NO_DATA = re.compile(r"case\s+\{([^}]*)\}\s*\n\s*error\(\[name ' - no data available")


def main(spinach_root: str) -> None:
    root = Path(spinach_root)
    source = (root / "kernel" / "spin.m").read_text(encoding="utf-8")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    entries = ENTRY.findall(source)
    names = [name for name, _, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate isotope names in spin.m")
    n_cases = source.count("case '")
    if n_cases != len(entries):
        raise ValueError(f"parsed {len(entries)} of {n_cases} single-name case entries")
    (no_data,) = NO_DATA.findall(source)
    missing = re.findall(r"'([^']+)'", no_data)

    lines = [
        f'"""Isotope table generated from Spinach kernel/spin.m at commit {commit}.',
        "",
        "Do not edit by hand; regenerate with porting/tools/gen_isotopes.py.",
        "Values: name -> (multiplicity, magnetogyric ratio in rad/(s*T)).",
        '"""',
        "",
        "ISOTOPES: dict[str, tuple[int, float]] = {",
    ]
    lines += [f"    {name!r}: ({int(mult)}, {gamma.strip()}),"
              for name, mult, gamma in entries]
    lines += ["}", "", f"NO_DATA: frozenset[str] = frozenset({sorted(missing)!r})", ""]
    out = Path(__file__).resolve().parents[2] / "src" / "respinach" / "_isotope_data.py"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(entries)} isotopes and {len(missing)} no-data names to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
