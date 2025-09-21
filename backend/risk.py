import numpy as np
from typing import List, Dict
from .propagation import ECIState, min_distance_and_rel_speed


def risk_score_from_distance(
    d_km: float,
    vrel_kms: float,
    threshold_km: float = 6500.0,   # used as the distance scale d0
    v0_kms: float = 7.5,            # speed scale (~LEO relative speed)
) -> float:
    """
    Smooth risk in [0,1]:
      proximity = 1 / (1 + (d/d0)^2)
      speed     = tanh(v_rel / v0)
      risk      = proximity * speed

    This never hard-caps to 0 at large d; at d == d0 you get proximity=0.5.
    """
    d0 = max(1e-6, float(threshold_km))
    d   = max(0.0, float(d_km))
    v   = max(0.0, float(vrel_kms))

    proximity = 1.0 / (1.0 + (d / d0) ** 2)
    speed_fac = np.tanh(v / float(v0_kms))
    s = proximity * speed_fac
    return float(np.clip(s, 0.0, 1.0))


def compute_close_approaches(
    debris: List[ECIState],
    targets: Dict[str, List[ECIState]],
    threshold_km: float,
):
    results = []
    for name, states in targets.items():
        dmin, vrel, tmin = min_distance_and_rel_speed(debris, states)
        # threshold_km now acts as the "distance scale" (d0)
        score = risk_score_from_distance(dmin, vrel, threshold_km)
        results.append(
            dict(
                target=name,
                min_distance_km=dmin,
                timestamp_min=tmin,
                rel_speed_kms=vrel,
                risk_score=score,
            )
        )
    # Highest score first; tie-breaker prefers smaller distance
    results.sort(key=lambda x: (x["risk_score"], -x["min_distance_km"]), reverse=True)
    return results
