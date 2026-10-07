# Research figure captions

**Learning.** Validation pixel F1 before post-training temperature/threshold calibration, over ten epochs. Each variant uses 752 training examples and 176 validation examples; one seed (42), no multi-run interval. Checkpoint selection uses validation only.

**Detectors.** Pooled pixel F1 for five frozen detectors. Bars are 95% percentile confidence intervals from 1000 nonparametric bootstrap replicates. Simulation uses 128 mixed and 64 stronger-nuisance independent scenes. Real data uses 20 held-out panoramas, grouping four crops per panorama. Intervals quantify test-sample variation, not training randomness; geographic independence is unverified.

**Policy.** Measured Brier-error reduction over 128 independent simulator test scenes. Each strategy is evaluated on the same scenes. Points show all scene gains; diamonds are means; boxes show medians and IQR, whiskers extend to 1.5 IQR. Uniform is its exact per-scene expected gain over three actions. The fixed action uses validation-only selection; the quality heuristic is predeclared. Passive has zero gain. Oracle uses ground truth and is unattainable at inference. Negative values mean the chosen acquisition made error worse. Abstention chooses passive when all predicted gains are nonpositive. No significance symbols are used.
