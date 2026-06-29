"""LCF flow layer — wraps each pipeline behind one common interface.

A Flow is a named unit of work with a cadence that, given the shared
LCFRuntime, runs and returns a standard FlowResult. The registry lists all
flows so the dispatcher (lcf.py) and scheduler can run any of them uniformly.
"""

from .base import Flow, FlowResult, Decision
from .registry import all_flows, get_flow

__all__ = ["Flow", "FlowResult", "Decision", "all_flows", "get_flow"]
