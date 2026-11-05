---
license: other
pretty_name: SSU Data Dash 2026
tags: [tabular-regression, real-estate, synthetic]
---

# SSU Data Dash 2026: Toronto home prices

Synthetic data for the SSU Data Dash case competition. No real listing, address or price appears.
Competition use only. Code, rules and submission format:
[github.com/erwinzhang7/SSU-Data-Dash](https://github.com/erwinzhang7/SSU-Data-Dash).

| File | Rows | What it is |
|---|---|---|
| `listings` | 150,000 | Sales, January 2021 to May 2026, with sold prices. Train on these. |
| `test_off_market` | 3,974 | Homes to value before listing: no list price. |
| `test_list_aware` | 4,064 | Listed homes with their list price: predict the sale price. |

Each comes as `.parquet` and `.csv`. Test homes sold June to September 2026; their prices are held back.

**Columns:** `ml_num` (id), `sold_price` (target), `list_price`, `date_listed`, `date_sold`,
`days_on_market`, `house_type_name`, `house_style`, `bedrooms`, `washrooms`, `parking_total`,
`garage`, `house_area` (sq ft, sometimes a range), `lot_front`, `lot_depth` (ft), `municipality`,
`community`, `latitude`, `longitude`, `external_avm_estimate` and `external_avm_estimate_date`
(a third-party estimate and when it was made), `remarks` (description, written by a language model).
