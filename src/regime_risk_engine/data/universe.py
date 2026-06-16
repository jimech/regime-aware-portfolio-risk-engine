from dataclasses import dataclass
from typing import Any

MIN_ASSET_COUNT = 15
MAX_ASSET_COUNT = 25

REQUIRED_ASSET_FIELDS = {
    "ticker",
    "name",
    "asset_class",
    "region",
    "role",
}


class AssetUniverseError(ValueError):
    """Raised when the asset universe is invalid."""


@dataclass(frozen=True, slots=True)
class Asset:
    """Metadata for one investable asset in the project universe."""

    ticker: str
    name: str
    asset_class: str
    region: str
    role: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Asset":
        """Create an Asset from a configuration dictionary."""
        missing_fields = REQUIRED_ASSET_FIELDS.difference(data)

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise AssetUniverseError(f"Missing required asset field(s): {missing}")

        ticker = data["ticker"]
        name = data["name"]
        asset_class = data["asset_class"]
        region = data["region"]
        role = data["role"]

        values = {
            "ticker": ticker,
            "name": name,
            "asset_class": asset_class,
            "region": region,
            "role": role,
        }

        for field_name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise AssetUniverseError(
                    f"Asset field '{field_name}' must be a non-empty string"
                )

        return cls(
            ticker=ticker.upper(),
            name=name,
            asset_class=asset_class,
            region=region,
            role=role,
        )


def load_asset_universe(config: dict[str, Any]) -> list[Asset]:
    """Load the asset universe from a project configuration dictionary."""
    assets_section = config.get("assets")

    if not isinstance(assets_section, dict):
        raise AssetUniverseError("Config must contain an 'assets' section")

    universe_data = assets_section.get("universe")

    if not isinstance(universe_data, list):
        raise AssetUniverseError("Config must contain assets.universe as a list")

    assets: list[Asset] = []

    for asset_data in universe_data:
        if not isinstance(asset_data, dict):
            raise AssetUniverseError("Each asset entry must be a dictionary")

        assets.append(Asset.from_dict(asset_data))

    validate_asset_universe(assets)

    return assets


def validate_asset_universe(assets: list[Asset]) -> None:
    """Validate the complete asset universe."""
    asset_count = len(assets)

    if asset_count < MIN_ASSET_COUNT:
        raise AssetUniverseError(
            f"Asset universe must contain at least {MIN_ASSET_COUNT} assets"
        )

    if asset_count > MAX_ASSET_COUNT:
        raise AssetUniverseError(
            f"Asset universe must contain no more than {MAX_ASSET_COUNT} assets"
        )

    tickers = [asset.ticker for asset in assets]
    duplicate_tickers = sorted(
        {ticker for ticker in tickers if tickers.count(ticker) > 1}
    )

    if duplicate_tickers:
        duplicates = ", ".join(duplicate_tickers)
        raise AssetUniverseError(f"Duplicate asset ticker(s): {duplicates}")


def get_tickers(assets: list[Asset]) -> list[str]:
    """Return tickers from an asset universe."""
    return [asset.ticker for asset in assets]


def group_assets_by_class(assets: list[Asset]) -> dict[str, list[Asset]]:
    """Group assets by asset class."""
    grouped: dict[str, list[Asset]] = {}

    for asset in assets:
        grouped.setdefault(asset.asset_class, []).append(asset)

    return grouped
