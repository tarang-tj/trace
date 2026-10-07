import numpy as np
import pytest
import torch
from scipy.ndimage import shift

from tracecv.data import SimulationDataset, pcd_ids
from tracecv.interventions import build_lattice, estimate_translation
from tracecv.metrics import calibrate, score
from tracecv.model import TRACE, segmentation_loss
from tracecv.simulator import ACTIONS, make_scene


def test_reobservation_preserves_scene_and_label():
    scene = make_scene(123)
    mask, latent = scene.mask.copy(), scene.latent.copy()
    for action in ACTIONS:
        observation, valid = scene.observe(action)
        assert observation.shape == scene.reference.shape
        assert valid.sum() > 0
        assert np.array_equal(scene.mask, mask)
        assert np.array_equal(scene.latent, latent)
    a, v = scene.observe()
    b, w = scene.observe()
    assert np.array_equal(a, b) and np.array_equal(v, w)


@pytest.mark.parametrize("scenario", ["lighting", "camera", "blur", "clean"])
def test_sensor_nuisance_is_not_labeled_physical_change(scenario):
    scene = make_scene(12, scenario=scenario)
    assert scene.mask.sum() == 0
    assert np.array_equal(scene.reference, scene.latent)


def test_registration_recovers_translation_and_reduces_residual():
    a = make_scene(15, scenario="clean").reference
    b = shift(a, (2, -3, 0), mode="nearest")
    assert estimate_translation(a, b) == (-2, 3)
    vertices, _ = build_lattice(a, b)
    inner = (slice(7, -7), slice(7, -7))
    assert np.abs(a[inner] - vertices[2][inner]).mean() < .001
    assert np.abs(a[inner] - b[inner]).mean() > .01


def test_model_loss_and_trust_have_valid_gradients():
    torch.set_num_threads(2)
    a, v, y, valid = SimulationDataset(1)[0]
    model = TRACE()
    output = model(a[None], v[None])
    assert output["logits"].shape == (1, 1, 96, 96)
    assert torch.allclose(output["weights"].sum(1), torch.ones(1, 96, 96), atol=1e-6)
    loss = segmentation_loss(output, y[None], valid[None])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pyramid.encoder[0][0].weight.grad.abs().sum() > 0
    assert model.survival[-1].weight.grad.abs().sum() > 0


def test_validation_calibration_is_finite_and_improves_nll():
    logits = np.array([-5, -4, 2, 3, -3, 1, 4, 5], np.float32)
    target = np.array([0, 0, 1, 1, 1, 0, 1, 0], np.float32)
    result = calibrate(logits, target)
    def nll(t):
        return (np.logaddexp(0, logits / t) - target * logits / t).mean()
    assert nll(result["temperature"]) <= nll(1) + 1e-5
    assert 0 < result["threshold"] < 1
    assert score([0, 1], [0, 1])["f1"] == 1


def test_real_split_is_by_panorama_and_disjoint():
    from pathlib import Path
    if not Path("data/pcd/provenance.json").exists():
        pytest.skip("Optional research dataset not downloaded")
    splits = [set(pcd_ids("data/pcd", split)) for split in ["train", "validation", "test"]]
    assert [len(s) for s in splits] == [60, 20, 20]
    assert len(set.union(*splits)) == 100
    assert not any(splits[i] & splits[j] for i in range(3) for j in range(i))
