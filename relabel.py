"""Backward-compatible entrypoint for historical relabeling.

Prefer:
    python -m ml.relabel ...

This wrapper keeps old command compatibility:
    python relabel.py ...
"""

from ml.relabel import main


if __name__ == "__main__":
    main()
