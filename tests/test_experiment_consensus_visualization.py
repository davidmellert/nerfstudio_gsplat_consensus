from nerfstudio.scripts.experiment import _build_trainer_config


def test_experiment_config_maps_consensus_visualization():
    config = _build_trainer_config(
        {
            "method": "splatfacto",
            "name": "viz-test",
            "consensus": {
                "mode": "online",
                "trainable": ["features_dc", "features_rest", "opacities"],
                "visualization": {
                    "enabled": True,
                    "interval": 25,
                    "window": 7,
                    "output_dir": "viz_out",
                    "groups": ["features_dc", "opacities"],
                    "save_png": False,
                    "save_npz": True,
                },
            },
        },
        suite_name=None,
        run_name="viz-test",
    )

    model_config = config.pipeline.model
    assert model_config.gaussian_consensus_visualization_enabled is True
    assert model_config.gaussian_consensus_visualization_interval == 25
    assert model_config.gaussian_consensus_visualization_window == 7
    assert model_config.gaussian_consensus_visualization_output_dir == "viz_out"
    assert model_config.gaussian_consensus_visualization_groups == ("features_dc", "opacities")
    assert model_config.gaussian_consensus_visualization_save_png is False
    assert model_config.gaussian_consensus_visualization_save_npz is True


def test_experiment_config_empty_visualization_groups_uses_default():
    config = _build_trainer_config(
        {
            "method": "splatfacto",
            "name": "viz-test",
            "consensus": {
                "mode": "online",
                "trainable": "appearance",
                "visualization": {"enabled": True, "groups": []},
            },
        },
        suite_name=None,
        run_name="viz-test",
    )

    assert config.pipeline.model.gaussian_consensus_visualization_groups == ()
