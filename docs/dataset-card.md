# Data and label protocol

## Procedural paired scenes

The renderer creates textured surfaces, six colored parts, and added/removed/moved-object labels in
reference coordinates. Labels represent changed object supports, including the interior of rings.
Lighting gain/bias/shadows, translation, blur, and noise independently alter acquisition. Sensor-only
scenarios have zero physical-change labels. Re-observation preserves scene geometry and labels.
Moving a part can overlap another part; this is a limited procedural distribution, not an industrial
physics simulator. Held-out seeds prevent identical generated scenes; they do not create a new class
of objects. OOD doubles nuisance severity but retains the same renderer family.

## PCD TSUNAMI

Author source: [PCD dataset](https://sakuradaken.net/pcd_dataset.html), associated with
[BMVC 2015 scene-change detection](https://www.bmva-archive.org.uk/bmvc/2015/papers/paper061/index.html).
The compact TSUNAMI set has 100 panoramic image pairs and author masks. Downloaded raw archive hash:
`3589bd3f33209bc197c727d1541ec1a8a55650f6ffadcda5c39f76b28968f911`.

White means annotated change. Some provided masks contain gray boundaries; binarize at 128 before
training or scoring. Split entire sorted panoramas using RNG seed 731 into 60 train, 20 validation,
20 test, then take four horizontal crops per panorama and resize to 96 × 96. This prevents crop-level
leakage. It does not guarantee geographically disjoint scenes or rule out nearby panoramic overlap.
The exported manifest makes this limitation auditable. Five boundary pixels are excluded from scoring.
This custom split/resolution is not an official benchmark protocol; never compare these scores to a
published ranking. The dataset's site recommends the newer PSCD benchmark.

Author terms permit research use and retain copyright. Raw data is downloaded locally and excluded
from this repository. The code license does not grant rights to the data or authorize commercial use
of learned weights. Inspect the author terms before any distribution or deployment of the checkpoint.
The initial checkpoint is a local research artifact, not a commercial release.

## Field capture experiment

Use at least 20 distinct surfaces/locations, physical changes and unchanged controls, repeated
lighting/pose/focus states, and independently annotated masks. Split by location before capture-derived
crops. Record calibration targets and the costs of each acquisition action. Freeze the model and
decision threshold before test capture. Report segmentation, false positives on unchanged scenes,
calibration, selective risk, action regret, latency, and all failures. This experiment is specified
but has not been performed by this project.
