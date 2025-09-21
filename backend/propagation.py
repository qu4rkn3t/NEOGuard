from sgp4.api import Satrec
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ECIState:
    t: float
    r: np.ndarray
    v: np.ndarray


def propagate_tle(line1: str, line2: str, minutes: int) -> List[ECIState]:
    sat = Satrec.twoline2rv(line1, line2)
    jd = sat.jdsatepoch
    fr = sat.jdsatepochF
    states: List[ECIState] = []
    for m in range(minutes + 1):
        jd_step = jd + (fr + m / (24.0 * 60.0))
        e, r, v = sat.sgp4(jd_step, 0.0)
        if e != 0:
            continue
        states.append(ECIState(t=float(m * 60.0), r=np.array(r), v=np.array(v)))
    return states


def min_distance_and_rel_speed(
    a: List[ECIState], b: List[ECIState]
) -> Tuple[float, float, float]:
    dists = []
    vrels = []
    times = []
    for sa, sb in zip(a, b):
        dr = sa.r - sb.r
        dv = sa.v - sb.v
        d = float(np.linalg.norm(dr))
        vrel = float(np.linalg.norm(dv))
        dists.append(d)
        vrels.append(vrel)
        times.append(sa.t)
    idx = int(np.argmin(dists))
    return dists[idx], vrels[idx], times[idx]
