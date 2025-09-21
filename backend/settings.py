from pydantic import BaseModel


class Settings(BaseModel):
    nasa_api_key: str = "kxzkc6cT0B7cVo2K6zhybV9tqFTHfY85ofBShxsz"
    nasa_base_url: str = "https://api.nasa.gov"
    nasa_tle_path: str = "/tle"

    allow_tle_fallback: bool = True
    celestrak_tle_url: str = ("https://celestrak.org/NORAD/elements/gp.php")
    cors_allow_origins: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    cors_allow_methods: list[str] = ["*"]


settings = Settings()
