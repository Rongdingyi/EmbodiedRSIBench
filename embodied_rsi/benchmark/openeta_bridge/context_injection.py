"""Compatibility shim: legacy OpenETA track re-exports the canonical RSI context.

The canonical implementation lives in `benchmark.rsi.context`. This module is
kept only so the historical OpenETA substrate (Pilot-60 v0.7) stays reproducible;
do not import it from canonical agent code paths.
"""
from benchmark.rsi.context import *  # noqa: F401,F403
