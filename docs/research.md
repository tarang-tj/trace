# Research positioning

TRACE is a proposed research technology, not a verified first invention. Its falsifiable claim is that
pixelwise learned survival/trust pooling across explicit acquisition probes improves physical-change
segmentation, and that observed disagreement helps a learned policy choose valuable re-observations.

| Prior work | Established capability | TRACE experiment |
|---|---|---|
| [Segment Any Change, 2024](https://arxiv.org/abs/2402.01188) | Training-free bitemporal change matching with foundation-model features | Small trained model; explicit photometric/registration lattice and learned trust |
| [Dual Task Learning, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Dual_Task_Learning_by_Leveraging_Both_Dense_Correspondence_and_Mis-Correspondence_CVPR_2022_paper.pdf) | Joint dense correspondence and change detection with imperfect matches | Fixed label-free probes followed by shared branch supervision; correspondence learning is not claimed new |
| [ChangeSim, 2021](https://arxiv.org/abs/2103.05368) | Industrial scene-change simulation and change-type labels | Compact simulator independently intervenes on acquisition nuisance and keeps labels fixed |
| [Active Robot Vision, 2023](https://arxiv.org/abs/2307.14105) | Active object-change inspection and lightweight planning simulation | Action targets are measured calibrated Brier reduction on independent sensor re-observations |
| [Active Damage Segmentation, 2023](https://www.dpi-proceedings.com/index.php/shm2023/article/download/36974/35549) | Active lighting/camera control for segmentation | Relighting and viewpoint/focus choice already have precedent; compare planner against random and oracle |
| [RUAC, 2026](https://proceedings.mlr.press/v306/zhou26ar.html) | Robustness and uncertainty alignment for segmentation shifts | Error head is an audited baseline component; uncertainty estimation is not claimed new |

The distinctive implementation is the combined intervention lattice, learned pixelwise pooling,
calibration, and empirically supervised follow-up loop. Literature search does not establish global
novelty. A publishable claim needs stronger models, contemporary benchmarks, multiple training seeds,
independent physical captures, and an expert novelty review.

## Challenge the premise

The most fragile assumption is that inexpensive computational transformations preserve physical
change while removing sensor effects. Registration can erase a moved object, and quantile correction
can suppress a real material/color change. Conventional alternatives are a corrected Siamese model
or foundation-model matcher. A constraint reversal is to control acquisition so nuisance disappears
before inference. A plausible alternative is to deploy disagreement as an operator inspection trigger
rather than attempt automated diagnosis. Prefer TRACE only if its added computation improves measured
held-out error and follow-up value; reject the pooling or action policy if simpler baselines match it.

The included ablations directly test the first decision, and passive/uniform/oracle planner evaluation
tests the second. The current small study cannot establish superiority over large pretrained systems.
Negative results must stay visible in the model card and studio.
