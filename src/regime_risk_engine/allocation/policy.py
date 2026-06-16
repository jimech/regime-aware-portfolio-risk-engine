from collections.abc import Mapping

import pandas as pd


class RegimeAllocationPolicyError(ValueError):
    """Raised when regime allocation policy cannot be used."""


RegimeWeightPolicy = Mapping[int, Mapping[str, float]]


def validate_regime_allocation_policy(
    policy: RegimeWeightPolicy,
    available_tickers: list[str],
    fallback_weights: Mapping[str, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Validate regime-to-weight allocation policy.

    Args:
        policy: Mapping from numeric regime to ticker weights.
        available_tickers: Tickers available in the return universe.
        fallback_weights: Optional fallback weights for unknown regimes.

    Returns:
        Cleaned policy with uppercase tickers and float weights.
    """
    if not policy:
        raise RegimeAllocationPolicyError("Regime allocation policy cannot be empty")

    clean_tickers = [str(ticker).upper().strip() for ticker in available_tickers]

    if not clean_tickers:
        raise RegimeAllocationPolicyError("Available tickers cannot be empty")

    cleaned_policy: dict[int, dict[str, float]] = {}

    for regime, weights in policy.items():
        cleaned_policy[int(regime)] = validate_target_weights(
            weights=weights,
            available_tickers=clean_tickers,
        )

    if fallback_weights is not None:
        validate_target_weights(
            weights=fallback_weights,
            available_tickers=clean_tickers,
        )

    return cleaned_policy


def validate_target_weights(
    weights: Mapping[str, float],
    available_tickers: list[str],
) -> dict[str, float]:
    """Validate one target allocation."""
    if not weights:
        raise RegimeAllocationPolicyError("Target weights cannot be empty")

    clean_tickers = [str(ticker).upper().strip() for ticker in available_tickers]
    available_ticker_set = set(clean_tickers)

    clean_weights = {
        str(ticker).upper().strip(): float(weight) for ticker, weight in weights.items()
    }

    weight_tickers = set(clean_weights)

    missing_tickers = sorted(available_ticker_set.difference(weight_tickers))
    extra_tickers = sorted(weight_tickers.difference(available_ticker_set))

    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise RegimeAllocationPolicyError(f"Missing weight(s) for ticker(s): {missing}")

    if extra_tickers:
        extra = ", ".join(extra_tickers)
        raise RegimeAllocationPolicyError(
            f"Weight provided for unknown ticker(s): {extra}"
        )

    if any(weight < 0 for weight in clean_weights.values()):
        raise RegimeAllocationPolicyError("Target weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > 1e-8:
        raise RegimeAllocationPolicyError("Target weights must sum to 1.0")

    return clean_weights


def get_target_weights_for_regime(
    regime: int,
    policy: RegimeWeightPolicy,
    available_tickers: list[str],
    fallback_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Get target weights for one regime."""
    cleaned_policy = validate_regime_allocation_policy(
        policy=policy,
        available_tickers=available_tickers,
        fallback_weights=fallback_weights,
    )

    clean_regime = int(regime)

    if clean_regime in cleaned_policy:
        return cleaned_policy[clean_regime]

    if fallback_weights is not None:
        return validate_target_weights(
            weights=fallback_weights,
            available_tickers=available_tickers,
        )

    raise RegimeAllocationPolicyError(
        f"No allocation policy found for regime: {clean_regime}"
    )


def build_regime_target_weight_frame(
    regime_labels: pd.Series,
    policy: RegimeWeightPolicy,
    available_tickers: list[str],
    fallback_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build date-indexed target weights from regime labels."""
    labels = _validate_regime_labels(regime_labels)
    cleaned_policy = validate_regime_allocation_policy(
        policy=policy,
        available_tickers=available_tickers,
        fallback_weights=fallback_weights,
    )

    clean_available_tickers = [
        str(ticker).upper().strip() for ticker in available_tickers
    ]

    fallback = None

    if fallback_weights is not None:
        fallback = validate_target_weights(
            weights=fallback_weights,
            available_tickers=clean_available_tickers,
        )

    rows: list[dict[str, float]] = []

    for regime in labels:
        clean_regime = int(regime)

        if clean_regime in cleaned_policy:
            rows.append(cleaned_policy[clean_regime])
        elif fallback is not None:
            rows.append(fallback)
        else:
            raise RegimeAllocationPolicyError(
                f"No allocation policy found for regime: {clean_regime}"
            )

    weight_frame = pd.DataFrame(
        rows,
        index=labels.index,
        columns=clean_available_tickers,
    )
    weight_frame.index.name = "date"

    return weight_frame.sort_index(axis=1)


def build_equal_weight_fallback(
    available_tickers: list[str],
) -> dict[str, float]:
    """Build equal-weight fallback allocation for available tickers."""
    clean_tickers = [str(ticker).upper().strip() for ticker in available_tickers]

    if not clean_tickers:
        raise RegimeAllocationPolicyError("Available tickers cannot be empty")

    duplicate_tickers = sorted(
        {ticker for ticker in clean_tickers if clean_tickers.count(ticker) > 1}
    )

    if duplicate_tickers:
        duplicates = ", ".join(duplicate_tickers)
        raise RegimeAllocationPolicyError(f"Duplicate ticker(s): {duplicates}")

    weight = 1.0 / len(clean_tickers)

    return {ticker: weight for ticker in clean_tickers}


def _validate_regime_labels(regime_labels: pd.Series) -> pd.Series:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeAllocationPolicyError("Regime labels index must be a DatetimeIndex")

    if regime_labels.empty:
        raise RegimeAllocationPolicyError("Regime labels are empty")

    if regime_labels.index.has_duplicates:
        raise RegimeAllocationPolicyError("Regime labels contain duplicate dates")

    if regime_labels.isna().any():
        raise RegimeAllocationPolicyError("Regime labels contain missing values")

    labels = regime_labels.copy()
    labels.index = pd.to_datetime(labels.index)

    return labels.sort_index()
