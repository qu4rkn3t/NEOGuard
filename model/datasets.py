import torch, httpx, numpy as np, os
from sgp4.api import Satrec


def _parse_celestrak_tle(text: str) -> tuple[str, str, str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Invalid TLE text")
    if lines[0].startswith("1 ") and lines[1].startswith("2 "):
        name = "UNKNOWN"
        line1, line2 = lines[0], lines[1]
    else:
        name = lines[0]
        line1, line2 = lines[1], lines[2]
    return name, line1, line2


def fetch_tle_json(norad: int, api_key: str) -> dict:
    nasa_base = os.getenv("NASA_BASE_URL", "https://api.nasa.gov")
    nasa_path = os.getenv("NASA_TLE_PATH", "/tle")
    allow_fallback = os.getenv("NASA_ALLOW_FALLBACK", "true").lower() == "true"
    try:
        url = f"{nasa_base}{nasa_path}"
        params = {"api_key": api_key, "NORAD": str(norad)}
        r = httpx.get(url, params=params, timeout=20.0)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "member" in data:
                data = data["member"][0]
            name = data.get("NAME") or data.get("name") or f"NORAD-{norad}"
            line1 = (
                data.get("TLE_LINE1")
                or data.get("line1")
                or data.get("tle", {}).get("line1")
            )
            line2 = (
                data.get("TLE_LINE2")
                or data.get("line2")
                or data.get("tle", {}).get("line2")
            )
            if line1 and line2:
                return {"name": name, "line1": line1, "line2": line2}
    except Exception:
        pass

    if not allow_fallback:
        raise RuntimeError("NASA TLE endpoint unavailable and fallback disabled")

    ct_url = os.getenv(
        "CELESTRAK_TLE_URL", "https://celestrak.org/NORAD/elements/gp.php"
    )
    r = httpx.get(ct_url, params={"CATNR": str(norad), "FORMAT": "TLE"}, timeout=20.0)
    r.raise_for_status()
    name, line1, line2 = _parse_celestrak_tle(r.text)
    return {"name": name, "line1": line1, "line2": line2}


def tle_to_sgp4(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2)


def make_baseline_sequence(sat: Satrec, minutes: int = 360):
    jd = sat.jdsatepoch
    fr = sat.jdsatepochF
    seq = []
    for m in range(minutes + 1):
        jd_step = jd + (fr + m / (24.0 * 60.0))
        e, r, v = sat.sgp4(jd_step, 0.0)
        if e == 0:
            seq.append((m * 60.0, np.array(r, dtype=float), np.array(v, dtype=float)))

    x = np.stack([np.concatenate([r, v]) for _, r, v in seq], axis=0)
    return x


def prepare_training_data(norad: int, api_key: str, minutes: int = 360):
    data = fetch_tle_json(norad, api_key)
    line1 = data["line1"]
    line2 = data["line2"]
    sat = tle_to_sgp4(line1, line2)
    x = make_baseline_sequence(sat, minutes=minutes)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
    line1 = data["line1"]
    line2 = data["line2"]
    sat = tle_to_sgp4(line1, line2)
    x = make_baseline_sequence(sat, minutes=minutes)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
