"""Structured fields extracted from historical used-car advertisements."""

import scrapy


class CarListing(scrapy.Item):
    brand = scrapy.Field()
    year = scrapy.Field()
    color = scrapy.Field()
    body = scrapy.Field()
    gear = scrapy.Field()
    price = scrapy.Field()
    usage = scrapy.Field()
    cit = scrapy.Field()
