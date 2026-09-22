"""Compatibility launcher for the full Crypto Low-Cap TRUE BREAKOUT PRO dashboard.

The full dashboard, including Live Deep Technical Analysis, is implemented in
crypto_lowcap_scanner_pro.py. This launcher keeps the older v8.4 entrypoint
working while ensuring it exposes the same full dashboard.
"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).with_name("crypto_lowcap_scanner_pro.py")),
    run_name="__main__",
)
