"""Opt-in historical Scrapy spider for used-car listings.

The extraction selectors match the original project and are NOT verified
against the current Divar website. Do not run without checking site policies.
"""

import scrapy

from car_price_prediction.crawler_urls import DEFAULT_CITIES, listing_urls
from divar_crawler.items import CarListing


class DivarSpider(scrapy.Spider):
    name = "Divar"
    allowed_domains = ["divar.ir"]

    def __init__(self, cities=None, car_model="pride/131", max_pages="2", **kwargs):
        super().__init__(**kwargs)
        self.cities = cities.split(",") if cities else DEFAULT_CITIES
        self.car_model = car_model
        self.max_pages = int(max_pages)

    def start_requests(self):
        # Only entrypoints generate paginated URLs; parse() must never recurse.
        for url in listing_urls(self.cities, self.car_model, self.max_pages):
            yield scrapy.Request(url, callback=self.parse)

    async def start(self):
        # Scrapy 2.13+ calls start(); older Scrapy versions call start_requests().
        for request in self.start_requests():
            yield request

    def parse(self, response):
        for href in response.css(".browse-post-list a::attr(href)").getall():
            yield response.follow(href, callback=self.parse_listing)

    def parse_listing(self, response):
        def field(position):
            return response.css(
                f"div.post-fields-item:nth-child({position}) > div:nth-child(2)::text"
            ).get(default="").strip()

        yield CarListing(
            brand=field(4),
            year=field(5),
            color=field(8),
            body=field(9),
            gear=field(10),
            price=field(11),
            usage=field(12),
            cit=response.css("a.post-fields-item__value:nth-child(2)::text")
            .get(default="").strip(),
        )
