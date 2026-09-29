---
crop: sugarcane
state: haryana
district: null
season: rabi
valid_from: "2024-01-05"
valid_until: null
source_authority: "ICAR - Indian Institute of Sugarcane Research (ICAR-IISR), Lucknow"
source_doc: "IISR Package of Practices for Sugarcane, 2023 edition"
source_url: "https://icar-iisr.res.in/publications"
schema_version: "2.0"
version: 3
decision_type: irrigation_rule
rule_id: SUGAR-IRR-014
language: en
---
# Irrigation Scheduling, Sugarcane, Grand Growth Phase (Haryana)

## When to irrigate
Trigger irrigation when the IMD 7-day accumulated rainfall falls short of district reference
evapotranspiration (ET0) and soil moisture in the 0-60 cm profile drops toward 40% of available
water capacity.

## Recommended volume and interval
- Apply about 75 mm per irrigation during grand growth.
- Typical interval on loam soils: every 7-10 days, adjusted by the IMD outlook.
- Shorten the interval on sandy-loam (drainage_class S); lengthen on clayey (M) soils.

## Heavy-rain / drainage caution
When an IMD Heavy-Rain-Warning is active on clayey soil (drainage_class M), SUSPEND irrigation and
clear furrow outlets — matched by Tier-1 rule SUGAR-M-HEAVYRAIN (action: clear furrow).

## Harvest-window rule
Suspend irrigation 2-3 weeks before harvest. Ties to Tier-1 rule SUGAR-HARVEST-CUTOFF.

## Sources
- ICAR-IISR Package of Practices for Sugarcane, 2023. pp. 21-24.
- IMD district forecast (runtime) used as the dynamic trigger only; the norm above is the fixed fact.
