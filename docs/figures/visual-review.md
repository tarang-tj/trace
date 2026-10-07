# Research figure visual review

All three final PNGs were opened with `view_image` at original resolution during the local review.

| Figure | Visual finding |
|---|---|
| Learning | Epoch points and three trajectories are readable. The lower-right legend avoids the data. Checkpoint markers remain distinguishable. One training seed is explicit; no uncertainty ribbon is invented. |
| Detectors | Three facets use the full 0–1 F1 scale. Model labels, distinct markers, and confidence intervals are clear. The real-data facet identifies 20 panorama units. The stronger-nuisance result does not conceal the corrected baseline's higher point estimate. |
| Policy | Every strategy's 128 scene values remain visible, including negative gains. Diamond means, IQR boxes, and 1.5-IQR whiskers are explained. Uniform random shows its exact expected value, and the oracle is identified as unattainable. |

No clipped labels, overlapping legends, or material rendering defect was observed. Automated layout
audits returned no issues for all three figures. Strict export checks passed for all six PNG/SVG
files; PNGs are approximately 300 DPI and SVG text remains editable. Numeric source CSVs and
[captions](captions.md) accompany the figures. [figure-audit.json](figure-audit.json) records data
profiles, resampling units, contracts, and layout results. These checks concern presentation, not
independent scientific validation; statistical and dataset limits remain in the model card.
