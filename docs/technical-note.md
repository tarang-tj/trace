# TRACE technical note

## Hypothesis and representation

Let \(R\) be a reference image and \(O\) a later observation. The supervision \(Y\) describes
physical object change in reference coordinates. Acquisition noise is not itself a positive label.
Define two image-only transforms: photometric alignment \(P_R\) and translation alignment \(T_R\).
The intervention lattice is

\[
\mathcal V(R,O)=\{O, P_R(O), T_R(O), P_R(T_R(O))\}.
\]

Photometric alignment matches channelwise 15th, 50th and 85th percentiles with bounded gain.
Translation alignment searches a ±4-pixel window using gradient mismatch, trimmed to reduce
the effect of changed regions. These transformations are hypotheses about nuisance, not guaranteed
physical interventions. The fourth vertex first registers and then aligns photometry.

A shared Siamese pyramid predicts branch logits \(e_k(x)\) and trust logits \(t_k(x)\).
Normalize trust across branches:

\[
w_k(x)=\operatorname{softmax}_k t_k(x),\qquad
z(x)=\sum_k w_k(x)e_k(x).
\]

The survival representation stacks weighted, mean, minimum, maximum, and variance logits:

\[
s(x)=[z(x),\;\operatorname{mean}_k e_k(x),\;\min_k e_k(x),\;\max_k e_k(x),\;
\operatorname{var}_k e_k(x)].
\]

A learned 1×1 residual network \(g\) produces \(\ell=z+g(s)\). The intuition is that persistence,
contradiction, and the spatial context of each corrected image jointly indicate change. This is
an empirical hypothesis: the network is free to learn a different statistical rule.

## Training and calibration

The segmentation objective combines valid-pixel binary cross entropy, per-image soft Dice, averaged
branch supervision, and learned error classification:

\[
L=L_{BCE}(\ell,Y)+L_{Dice}(\ell,Y)+0.25\,\operatorname{mean}_k L_{BCE}(e_k,Y)
+0.15\,L_{BCE}(h(\operatorname{stopgrad}s),\mathbf1[(\sigma(\ell)\ge0.5)\ne(Y\ge0.5)]).
\]

The error head \(h\) receives detached features so its loss does not alter the segmentation
representation. A temperature \(\tau\), fitted to validation NLL, produces \(p=\sigma(\ell/\tau)\).
A validation F1 sweep chooses the segmentation threshold. Error temperature is fitted separately
against mistakes at the selected threshold. Temperature fitting does not repair representation
errors or ensure calibration on shifted data. The error head's training and calibrated decision
thresholds differ; selective-risk measurements audit the consequence.

The model has a three-level encoder of widths 12, 24 and 48 with GroupNorm and SiLU. Features from
reference, observation, and their absolute difference are fused at every level. Bilinear upsampling
and skip fusion decode the branch maps. AdamW uses learning rate 0.002, weight decay 0.0001 and
gradient clipping at 5. The local run trains ten epochs at batch size eight, seed 42, selecting the
checkpoint by uncalibrated validation F1. See the executable code for exact valid masks and smoothing.

## Learn the value of acquiring another observation

For acquisition action \(a\), hold the latent scene fixed and render \(O_a\). Relight removes
lighting distortion, re-register removes sensor translation, and focus removes blur/noise.
Each action begins from the original sensor state; actions are independent rather than cumulative.
Measure gain on the same valid pixel intersection before and after:

\[
G_a=Brier(p(R,O),Y)-Brier(p(R,O_a),Y).
\]

A standardized twelve-feature vector \(d(R,O)\) contains estimated translation, residuals,
sharpness, probability summaries, error, branch disagreement and trust entropy. No latent state,
action outcome, or annotation enters this vector. A 12→32→32→3 MLP learns \(\widehat G_a\)
by squared error. Scaling training targets by 100 improves conditioning; saved outputs are
converted back into Brier units. Detector-disjoint scenes supply 256 training, 64 validation and
128 test examples. Best validation MSE selects weights over 250 epochs.

The ranking rule is \(\arg\max_a\widehat G_a\). The audit also examines abstaining when every
predicted gain is nonpositive. Passive, uniform expected gain, a validation-selected fixed action,
a predeclared quality heuristic, and the ground-truth oracle expose whether learning adds value.
The oracle is an upper bound, never an available deployment strategy.

## What would establish the claim

The current experiment tests the complete learned lattice against raw and corrected Siamese models
and two calibrated pixel-distance baselines. It does not separately isolate trust weighting,
residual statistics, branch supervision, or extra branch computation. A follow-on factorial ablation
should compare mean/min pooling, equal-compute ensembles and learned correspondence, and include
strong foundation-model matchers with identical data access.

The assumption most likely to fail is that inexpensive global corrections preserve physical
change while removing acquisition nuisance. A moved component can influence registration; a
changed material can influence quantile alignment. Evidence that a corrected single branch matches
TRACE under location-separated captures would favor that simpler model. Evidence that fixed
lighting control matches the policy would favor controlling acquisition directly.

A third option is to use probe disagreement only as an operator trigger. This gives up automated
action ranking but can be useful when hardware actions are expensive or poorly modeled. No variant
should receive a deployment recommendation from this small, single-seed, simulator-centered study.
The measured model card specifies exactly which conclusions the current evidence supports.
