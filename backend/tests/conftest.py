"""Test environment: fake credentials, temp dirs, watcher disabled.

Must run before backend.app is imported, hence module-level setup here.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="omnishot-test-"))

os.environ.setdefault("JINA_API_KEY", "test-key")
os.environ.setdefault("ES_URL", "http://localhost:9200")
os.environ["CHUNKS_DIR"] = str(_tmp / "chunks")
os.environ["WATCH_DIR"] = str(_tmp / "clips")
os.environ["OMNISHOT_DISABLE_WATCHER"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
