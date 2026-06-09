import torch

from nerfstudio.utils.consensus_visualization import compute_consensus_visualization_data


def test_consensus_visualization_aligned_views():
    view_grads = {
        "features_dc": [
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ]
    }
    final_grads = {"features_dc": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])}

    per_gaussian, scalar_attributes, rgb_attributes = compute_consensus_visualization_data(
        view_grads, final_grads, {"features_dc": 1.0}, eps=1e-8
    )

    assert torch.allclose(per_gaussian["mean_update_norm"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(per_gaussian["final_update_norm"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(per_gaussian["agreement"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(per_gaussian["disagreement"], torch.tensor([0.0, 0.0]))
    assert torch.equal(per_gaussian["dominant_view"], torch.tensor([0, -1]))
    assert "mean_update_norm" in scalar_attributes
    assert torch.allclose(rgb_attributes["rgb_update_mean"][0], torch.tensor([-1.0, 0.0, 0.0]))


def test_consensus_visualization_opposing_views_disagree():
    view_grads = {
        "features_dc": [
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
        ]
    }
    final_grads = {"features_dc": torch.zeros(1, 3)}

    per_gaussian, _, rgb_attributes = compute_consensus_visualization_data(
        view_grads, final_grads, {"features_dc": 1.0}, eps=1e-8
    )

    assert torch.allclose(per_gaussian["mean_update_norm"], torch.tensor([0.0]))
    assert torch.allclose(per_gaussian["agreement"], torch.tensor([0.0]))
    assert torch.allclose(per_gaussian["disagreement"], torch.tensor([1.0]))
    assert torch.allclose(rgb_attributes["rgb_update_mean"], torch.zeros(1, 3))


def test_consensus_visualization_dominant_view_and_opacity_direction():
    view_grads = {
        "opacities": [
            torch.tensor([[2.0]]),
            torch.tensor([[0.5]]),
        ]
    }
    final_grads = {"opacities": torch.tensor([[1.0]])}

    per_gaussian, scalar_attributes, _ = compute_consensus_visualization_data(
        view_grads, final_grads, {"opacities": 1.0}, eps=1e-8
    )

    assert torch.equal(per_gaussian["dominant_view"], torch.tensor([0]))
    assert torch.allclose(per_gaussian["dominance_strength"], torch.tensor([0.8]))
    assert torch.allclose(scalar_attributes["opacity_update_mean"], torch.tensor([-1.25]))
    assert torch.allclose(scalar_attributes["opacity_update_final"], torch.tensor([-1.0]))


def test_consensus_visualization_max_views_filters_to_largest_update():
    view_grads = {
        "features_dc": [
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[3.0, 0.0, 0.0]]),
        ]
    }
    final_grads = {"features_dc": torch.tensor([[3.0, 0.0, 0.0]])}

    per_gaussian, _, _ = compute_consensus_visualization_data(
        view_grads, final_grads, {"features_dc": 1.0}, eps=1e-8, max_views=1
    )

    assert torch.allclose(per_gaussian["view_update_norm"], torch.tensor([[0.0], [3.0]]))
    assert torch.equal(per_gaussian["dominant_view"], torch.tensor([1]))
    assert torch.allclose(per_gaussian["agreement"], torch.tensor([1.0]))
