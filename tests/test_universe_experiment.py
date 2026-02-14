from chromascope.experimental.universe import RESOLUTION_PRESETS, UNIVERSE_CONFIG


def test_universe_config_targets_new_style():
    assert UNIVERSE_CONFIG["style"] == "universe"
    assert UNIVERSE_CONFIG["dynamicBg"] is True
    assert UNIVERSE_CONFIG["bgParticles"] is True


def test_universe_has_4k_preset():
    assert RESOLUTION_PRESETS["4k"] == (3840, 2160)
