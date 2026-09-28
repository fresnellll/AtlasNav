from pathlib import Path

import pytest

from atlasnav.config import ConfigurationError, Pricing, load_model_profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "threshold", "currency"),
    [
        ("deepseek-v4-flash.toml", 2.75, "CNY"),
        ("mimo-v2.5.toml", 1.40, "CNY"),
        ("gpt-5.6-luna.toml", 0.65, "USD"),
        ("qwen-3.7-flash.toml", 1.95, "CNY"),
    ],
)
def test_paper_profiles(name: str, threshold: float, currency: str) -> None:
    value = load_model_profile(ROOT / "configs/paper" / name)
    assert value.safe_release.cost_threshold == threshold
    assert value.pricing.currency == currency
    assert value.safe_release.max_turns == 300
    assert value.safe_release.allow_tools is False


def test_pricing_uses_cache_as_separate_token_class() -> None:
    pricing = Pricing("CNY", 1.0, 0.2, 2.0)
    assert pricing.cost(
        input_tokens=1_000_000,
        cached_input_tokens=1_000_000,
        output_tokens=1_000_000,
    ) == pytest.approx(3.2)


def test_negative_tokens_rejected() -> None:
    pricing = Pricing("CNY", 1.0, 0.2, 2.0)
    with pytest.raises(ConfigurationError):
        pricing.cost(input_tokens=-1, cached_input_tokens=0, output_tokens=0)

