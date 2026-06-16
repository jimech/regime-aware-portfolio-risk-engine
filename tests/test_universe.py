import pytest

from regime_risk_engine.data.universe import (
    MIN_ASSET_COUNT,
    Asset,
    AssetUniverseError,
    get_tickers,
    group_assets_by_class,
    load_asset_universe,
    validate_asset_universe,
)


def make_asset(ticker: str, asset_class: str = "equity") -> dict[str, str]:
    return {
        "ticker": ticker,
        "name": f"{ticker} Fund",
        "asset_class": asset_class,
        "region": "us",
        "role": f"{ticker.lower()}_role",
    }


def make_valid_assets() -> list[dict[str, str]]:
    return [
        make_asset(f"A{i:02d}", "equity" if i < 8 else "fixed_income")
        for i in range(MIN_ASSET_COUNT)
    ]


def test_asset_from_dict_success() -> None:
    asset = Asset.from_dict(make_asset("spy"))

    assert asset.ticker == "SPY"
    assert asset.name == "spy Fund"
    assert asset.asset_class == "equity"
    assert asset.region == "us"
    assert asset.role == "spy_role"


def test_asset_from_dict_missing_field_raises_error() -> None:
    asset_data = make_asset("SPY")
    del asset_data["role"]

    with pytest.raises(AssetUniverseError, match="Missing required asset field"):
        Asset.from_dict(asset_data)


def test_load_asset_universe_success() -> None:
    config = {
        "assets": {
            "universe": make_valid_assets(),
        },
    }

    assets = load_asset_universe(config)

    assert len(assets) == MIN_ASSET_COUNT
    assert all(isinstance(asset, Asset) for asset in assets)


def test_validate_asset_universe_rejects_too_few_assets() -> None:
    assets = [Asset.from_dict(make_asset("SPY"))]

    with pytest.raises(AssetUniverseError, match="at least"):
        validate_asset_universe(assets)


def test_validate_asset_universe_rejects_duplicate_tickers() -> None:
    asset_data = make_valid_assets()
    asset_data[-1]["ticker"] = asset_data[0]["ticker"]

    assets = [Asset.from_dict(item) for item in asset_data]

    with pytest.raises(AssetUniverseError, match="Duplicate asset ticker"):
        validate_asset_universe(assets)


def test_get_tickers_returns_asset_tickers() -> None:
    assets = [
        Asset.from_dict(make_asset("SPY")),
        Asset.from_dict(make_asset("TLT", "fixed_income")),
    ]

    assert get_tickers(assets) == ["SPY", "TLT"]


def test_group_assets_by_class() -> None:
    assets = [
        Asset.from_dict(make_asset("SPY", "equity")),
        Asset.from_dict(make_asset("TLT", "fixed_income")),
        Asset.from_dict(make_asset("IEF", "fixed_income")),
    ]

    grouped = group_assets_by_class(assets)

    assert set(grouped) == {"equity", "fixed_income"}
    assert len(grouped["equity"]) == 1
    assert len(grouped["fixed_income"]) == 2
