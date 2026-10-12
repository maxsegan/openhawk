"""tennis_charting — the symbolic data layer.

Turns Match Charting Project notation into a rally-token dataset (+ per-player time index)
that the models in ``training/`` consume, and defines the shared shot vocabulary that CV
auto-charting output must also target.
"""

__all__ = ["mcp_parse", "tokenizer"]
