"""Point d'entrée de l'application Streamlit.

Usage :
    uv run python main.py            # lance l'application sur http://localhost:8501
    uv run python main.py -- --help  # options streamlit supplémentaires
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "app.py"
DEFAULT_PORT = "8501"


def main() -> None:
    """Lance app.py via l'API streamlit"""
    from streamlit.web import cli as stcli

    port = os.environ.get("PORT", DEFAULT_PORT)
    extra = [a for a in sys.argv[1:] if a != "--"]
    args = ["run", str(APP_FILE), "--server.port", port, *extra]

    print(f"▶ Système de recommandation d'animes : http://localhost:{port}")
    sys.exit(stcli.main(args=args, prog_name="streamlit"))


if __name__ == "__main__":
    main()
