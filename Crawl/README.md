# Historical Divar acquisition

This directory holds a **2022 snapshot**, its raw CSV data and optional Scrapy
crawler. It is not required for training on the versioned cleaned dataset.

The crawler has not been tested against the present website. Before using it,
review Divar's terms, robots policy and site structure. The old selectors are
fragile and may yield no records. Crawling is deliberately excluded from CI.

Install optional dependencies from the repository root:

```bash
python -m pip install -e ".[crawl]"
cd Crawl/divar_crawler
scrapy crawl Divar -a cities=tehran,karaj -a car_model=pride/131 -a max_pages=2 -O listings.csv
```

The spider now constructs each page URL once rather than recursively creating
another page range inside every response. Its page count and selected cities
are configurable, and the Scrapy project still respects robots.txt.

Offline data preparation can be run from the repository root:

```bash
python Crawl/divar_crawler/concate_cars_csv.py
car-price prepare --raw Crawl/divar_crawler/combined_csv.csv --output data/processed.csv
```

Use `car-price train --data data/processed.csv` to evaluate the new data.
Never assume a fresh crawl uses the same market distribution as the snapshot.
