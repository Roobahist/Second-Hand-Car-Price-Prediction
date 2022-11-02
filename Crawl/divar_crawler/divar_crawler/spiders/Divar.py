# -*- coding: utf-8 -*-
import scrapy

class AmazonItem(scrapy.Item):
    brand = scrapy.Field()
    year = scrapy.Field()
    color = scrapy.Field()
    body = scrapy.Field()
    gear = scrapy.Field()
    price = scrapy.Field()
    usage = scrapy.Field()
    cit = scrapy.Field()

class DivarSpider(scrapy.Spider):
    name = 'Divar'
    # intialize url
    start_urls = ['https://divar.ir/']


    def parse(self, response):
        # follow links to author pages
        for href in response.css('.browse-post-list a::attr(href)'):
            yield response.follow(href, self.parse_author)

        # cities we collect data
        shahr=['tehran','mashhad','karaj','shiraz','isfahan','tabriz','ahvaz','rasht','qom','kermanshah']

        # cars we collect data for them
        # cars=['pride/111','dena/basic','jac/s5','peugeot/206/5','peugeot/206/2',
            # 'peugeot/pars/basic','samand/lx','peugeot/405/glx-petrol','tiba/sedan','pride/131']

        start_urls=[]
        for city in shahr:
                start_urls.append('https://divar.ir/s/{}/car/{}'.format(city,'pride/131'))
        for i in range(0,200):
            for j in start_urls:
                  yield response.follow(j+'?page='+str(i), self.parse)
                  
    def parse_author(self, response):
        def extract_with_css(query):
            return response.css(query).get(default='').strip()

        item = AmazonItem()


        item['brand'] =  extract_with_css('div.post-fields-item:nth-child(4) > div:nth-child(2)::text') if  extract_with_css('div.post-fields-item:nth-child(4) > div:nth-child(2)::text')!= None else None
        item['year'] = extract_with_css('div.post-fields-item:nth-child(5) > div:nth-child(2)::text') if  extract_with_css('div.post-fields-item:nth-child(5) > div:nth-child(2)::text')!= None else None
        item['color'] = extract_with_css('div.post-fields-item:nth-child(8) > div:nth-child(2)::text') if extract_with_css('div.post-fields-item:nth-child(8) > div:nth-child(2)::text') != None else None
        item['body'] =extract_with_css('div.post-fields-item:nth-child(9) > div:nth-child(2)::text') if extract_with_css('div.post-fields-item:nth-child(9) > div:nth-child(2)::text') != None else None
        item['gear'] = extract_with_css('div.post-fields-item:nth-child(10) > div:nth-child(2)::text') if extract_with_css('div.post-fields-item:nth-child(10) > div:nth-child(2)::text') != None else None
        item['price'] =extract_with_css('div.post-fields-item:nth-child(11) > div:nth-child(2)::text') if extract_with_css('div.post-fields-item:nth-child(11) > div:nth-child(2)::text') != None else None
        item['usage'] =extract_with_css('div.post-fields-item:nth-child(12) > div:nth-child(2)::text') if extract_with_css('div.post-fields-item:nth-child(12) > div:nth-child(2)::text') != None else None
        item['cit'] = extract_with_css('a.post-fields-item__value:nth-child(2)::text') if extract_with_css('a.post-fields-item__value:nth-child(2)::text')!=None else None
        yield item