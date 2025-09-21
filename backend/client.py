import httpx
from .settings import settings
from typing import List, Dict, Any, Optional


def _parse_celestrak_tle(text: str) -> dict:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("Invalid TLE text")
    if lines[0].startswith("1 ") and lines[1].startswith("2 "):
        name = "UNKNOWN"
        line1, line2 = lines[0], lines[1]
    else:
        name = lines[0]
        line1, line2 = lines[1], lines[2]
    return {"name": name, "line1": line1, "line2": line2}


async def fetch_tle_by_norad(norad_id: int) -> dict:
    try:
        url = f"{settings.nasa_base_url}{settings.nasa_tle_path}"
        params = {"api_key": settings.nasa_api_key, "NORAD": str(norad_id)}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            items = []
            if isinstance(data, dict) and "member" in data:
                data = data["member"]
            if isinstance(data, dict) and "records" in data:
                data = data["records"]
            if isinstance(data, dict):
                data = [data]
            for item in data or []:
                name = item.get("NAME") or item.get("name") or f"NORAD-{norad_id}"
                line1 = (
                    item.get("TLE_LINE1")
                    or item.get("line1")
                    or item.get("tle", {}).get("line1")
                )
                line2 = (
                    item.get("TLE_LINE2")
                    or item.get("line2")
                    or item.get("tle", {}).get("line2")
                )
                if line1 and line2:
                    items.append({"name": name, "line1": line1, "line2": line2})
            if items:
                return {"records": items}
    except Exception:
        pass

    if not settings.allow_tle_fallback:
        raise httpx.HTTPStatusError(
            "NASA TLE endpoint unavailable and fallback disabled",
            request=None,
            response=None,
        )

    params = {"CATNR": str(norad_id), "FORMAT": "TLE"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(settings.celestrak_tle_url, params=params)
        r.raise_for_status()
        record = _parse_celestrak_tle(r.text)
        record["norad_id"] = norad_id
        return {"records": [record]}


async def fetch_latest_n_tles(limit: int = 50) -> dict:
    raise NotImplementedError(
        "Bulk fetch not wired; fetch by NORAD with fallback instead."
    )

CELESTRAK_GROUPS = {
    "fengyun1c": "1999-025",
    "iridium33": "iridium-33-debris",
    "cosmos1408": "cosmos-1408-debris",
}

async def fetch_celestrak_catalog(name: str, limit: int = 25,
                                  http: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    group = CELESTRAK_GROUPS.get(name.lower())
    if not group:
        raise ValueError(f"Unknown catalog '{name}'. Valid: {list(CELESTRAK_GROUPS.keys())}")

    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=TLE"

    owns_client = http is None
    http = http or httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "orbital-demo/1.0 (+local)"},
        follow_redirects=True,
    )
    try:
        resp = await http.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Upstream {url} -> {resp.status_code}: {resp.text[:200]}")

        lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
        records: List[Dict[str, Any]] = []

        i = 0
        while i + 2 < len(lines):
            nm, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            if l1.startswith("1 ") and l2.startswith("2 "):
                try:
                    norad = int(l1[2:7].strip())
                except Exception:
                    norad = None
                records.append({"name": nm, "norad_id": norad, "line1": l1, "line2": l2})
                if len(records) >= limit:
                    break
                i += 3
            else:
                i += 1

        return {"count": len(records), "records": records}
    finally:
        if owns_client:
            await http.aclose()
