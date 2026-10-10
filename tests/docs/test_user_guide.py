"""Run every Python code block of docs/user-guide.md, in order, in one namespace."""

import re
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "user-guide.md"


def test_user_guide_examples_run():
    blocks = re.findall(r"^```python\n(.*?)^```$", GUIDE.read_text(encoding="utf-8"), flags=re.S | re.M)
    assert len(blocks) >= 10
    namespace: dict = {}
    for n, block in enumerate(blocks):
        exec(compile(block, f"{GUIDE.name} block {n}", "exec"), namespace)
