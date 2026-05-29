#!/usr/bin/env python3
"""Rewrite README.md for the PyPI long_description.

GitHub renders the "How it works" loop as a native Mermaid diagram, but
PyPI has no Mermaid renderer and would show the raw fenced source on the
project page. This script - run *only* in the publish workflow against
the ephemeral build checkout - swaps the marked Mermaid block for the
equivalent SVG image, served from the public mirror's raw URL. The
committed README is left untouched, so the GitHub mirror keeps the live,
theme-aware diagram.

Fails loudly if the markers are missing, so a future README edit can't
silently ship raw Mermaid to PyPI.
"""

from __future__ import annotations

import pathlib
import re
import sys

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
START = "<!-- loop-diagram:mermaid"
END = "<!-- /loop-diagram:mermaid -->"

IMG = """<div align="center">

<img
  src="https://raw.githubusercontent.com/Artl13/robotrace-dev/main/assets/robotrace-loop.svg"
  alt="The RoboTrace loop: Record, Replay, Explain, then Verify and Evals by re-rolling a candidate policy against historical episodes - which feeds back into Record."
  width="820"
/>

</div>"""


def main() -> int:
    text = README.read_text(encoding="utf-8")
    # START opens an HTML comment with a trailing note, so match up to its
    # closing `-->` and then through the END marker.
    pattern = re.compile(
        re.escape(START) + r".*?-->.*?" + re.escape(END),
        re.DOTALL,
    )
    if not pattern.search(text):
        print(
            f"::error::loop-diagram markers not found in {README.name}; "
            "cannot build a PyPI-safe README. Re-add the "
            f"'{START} ... -->' / '{END}' markers around the Mermaid block.",
            file=sys.stderr,
        )
        return 1
    README.write_text(pattern.sub(IMG, text), encoding="utf-8")
    print(f"Rewrote {README.name}: Mermaid loop -> SVG image for PyPI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
