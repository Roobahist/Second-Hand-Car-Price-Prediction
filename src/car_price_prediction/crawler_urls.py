"""Pure URL configuration for the optional, historical Divar crawler."""

from collections.abc import Iterable
from urllib.parse import quote

DEFAULT_CITIES = (
    "tehran", "mashhad", "karaj", "shiraz", "isfahan", "tabriz",
    "ahvaz", "rasht", "qom", "kermanshah",
)


def listing_urls(
    cities: Iterable[str] = DEFAULT_CITIES,
    car_model: str = "pride/131",
    pages: int = 2,
) -> list[str]:
    """Return deterministic crawl entrypoints; no network requests are made."""
    if pages < 1 or pages > 200:
        raise ValueError("pages must be between 1 and 200")
    if not car_model or not all(
        segment.isascii() and segment.replace("-", "").isalnum()
        for segment in car_model.split("/")
    ):
        raise ValueError("car_model must contain slash-separated ASCII names")
    prepared = [city.strip().lower() for city in cities]
    if not prepared or any(not city or not city.isascii() or not city.isalpha()
                           for city in prepared):
        raise ValueError("cities must contain nonempty ASCII names")
    return [
        f"https://divar.ir/s/{quote(city, safe='')}/car/{car_model}?page={page}"
        for city in dict.fromkeys(prepared)
        for page in range(1, pages + 1)
    ]
