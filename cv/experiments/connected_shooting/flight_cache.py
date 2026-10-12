"""Bounded exact-value cache for repeated flight simulations within one fit.

Research optimization, not an artifact cache. Callers must key every simulator
input, including queries and physics configuration. Deep copies prevent results
or caller-owned arguments from mutating retained states. No rounding is used.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import numpy as np


def freeze(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple((k, freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return ("float64", np.float64(value).tobytes())
    return value


class FlightCache:
    """Per-flight LRU, owned by one solve and never shared with other attempts."""

    def __init__(self, entries_per_flight: int = 8):
        if type(entries_per_flight) is not int or entries_per_flight < 1:
            raise ValueError("positive cache capacity required")
        self.capacity = entries_per_flight
        self.rows = {}
        self.hits = self.misses = 0

    def simulate(self, index, simulator, *args, **kwargs):
        from cv.experiments.connected_shooting import passive_bounce

        key = passive_bounce.cache_prefix() + (simulator, freeze(args), freeze(kwargs))
        rows = self.rows.setdefault(index, OrderedDict())
        if key in rows:
            self.hits += 1
            rows.move_to_end(key)
            return deepcopy(rows[key])
        self.misses += 1
        result = simulator(*args, **kwargs)
        rows[key] = deepcopy(result)
        if len(rows) > self.capacity:
            rows.popitem(last=False)
        return result

    def receipt(self):
        return dict(
            hits=self.hits,
            misses=self.misses,
            entries_per_flight=self.capacity,
            retained_entries=sum(map(len, self.rows.values())),
            approximate_keys=False,
        )
