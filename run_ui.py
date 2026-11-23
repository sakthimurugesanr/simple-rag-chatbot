"""Launch the UI with ordinary Python or the bundled desktop Python runtime."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Desktop verification installs dependencies here, isolated from the global runtime.
local_dependencies = ROOT / ".test-deps"
if local_dependencies.is_dir():
    sys.path.insert(0, str(local_dependencies))
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "ui" / "streamlit_app.py"),
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    raise SystemExit(cli.main())
