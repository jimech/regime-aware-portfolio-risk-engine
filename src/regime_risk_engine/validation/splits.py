from dataclasses import dataclass

import pandas as pd


class TimeSeriesSplitError(ValueError):
    """Raised when time-series validation splits cannot be created."""


@dataclass(frozen=True, slots=True)
class TimeSeriesSplit:
    """Container for one chronological train/test split."""

    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def build_expanding_window_splits(
    dates: pd.DatetimeIndex,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    max_splits: int | None = None,
) -> list[TimeSeriesSplit]:
    """Build expanding-window train/test splits.

    The train window starts at the first date and expands over time.
    """
    clean_dates = _prepare_dates(dates)
    clean_step_size = _validate_split_parameters(
        train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
        max_splits=max_splits,
    )

    if initial_train_size + test_size > len(clean_dates):
        raise TimeSeriesSplitError(
            "Not enough dates for the requested expanding-window split"
        )

    splits: list[TimeSeriesSplit] = []
    test_start_position = initial_train_size
    split_id = 0

    while test_start_position + test_size <= len(clean_dates):
        train_dates = clean_dates[:test_start_position]
        test_dates = clean_dates[test_start_position : test_start_position + test_size]

        splits.append(
            _build_split(
                split_id=split_id,
                train_dates=train_dates,
                test_dates=test_dates,
            )
        )

        split_id += 1

        if max_splits is not None and split_id >= max_splits:
            break

        test_start_position += clean_step_size

    return splits


def build_rolling_window_splits(
    dates: pd.DatetimeIndex,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    max_splits: int | None = None,
) -> list[TimeSeriesSplit]:
    """Build rolling-window train/test splits.

    The train window keeps a fixed length and rolls forward over time.
    """
    clean_dates = _prepare_dates(dates)
    clean_step_size = _validate_split_parameters(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        max_splits=max_splits,
    )

    if train_size + test_size > len(clean_dates):
        raise TimeSeriesSplitError(
            "Not enough dates for the requested rolling-window split"
        )

    splits: list[TimeSeriesSplit] = []
    test_start_position = train_size
    split_id = 0

    while test_start_position + test_size <= len(clean_dates):
        train_start_position = test_start_position - train_size
        train_dates = clean_dates[train_start_position:test_start_position]
        test_dates = clean_dates[test_start_position : test_start_position + test_size]

        splits.append(
            _build_split(
                split_id=split_id,
                train_dates=train_dates,
                test_dates=test_dates,
            )
        )

        split_id += 1

        if max_splits is not None and split_id >= max_splits:
            break

        test_start_position += clean_step_size

    return splits


def split_frame_by_dates(
    data: pd.DataFrame,
    split: TimeSeriesSplit,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test frames using split dates.

    If date_col is omitted, the DataFrame must use a DatetimeIndex.
    If date_col is provided, that column is converted to datetime for matching.
    """
    if data.empty:
        raise TimeSeriesSplitError("Data frame cannot be empty")

    if date_col is None:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TimeSeriesSplitError("Data frame index must be a DatetimeIndex")

        if data.index.has_duplicates:
            raise TimeSeriesSplitError("Data frame index contains duplicate dates")

        train_frame = data.loc[data.index.isin(split.train_dates)].copy()
        test_frame = data.loc[data.index.isin(split.test_dates)].copy()

        return train_frame, test_frame

    if date_col not in data.columns:
        raise TimeSeriesSplitError(f"Date column not found: {date_col}")

    clean_data = data.copy()
    clean_data[date_col] = pd.to_datetime(clean_data[date_col])

    train_mask = clean_data[date_col].isin(split.train_dates)
    test_mask = clean_data[date_col].isin(split.test_dates)

    train_frame = clean_data.loc[train_mask].copy()
    test_frame = clean_data.loc[test_mask].copy()

    return train_frame, test_frame


def validate_time_series_split(split: TimeSeriesSplit) -> None:
    """Validate chronological split ordering."""
    if split.train_dates.empty:
        raise TimeSeriesSplitError("Train dates cannot be empty")

    if split.test_dates.empty:
        raise TimeSeriesSplitError("Test dates cannot be empty")

    if split.train_dates.has_duplicates:
        raise TimeSeriesSplitError("Train dates contain duplicates")

    if split.test_dates.has_duplicates:
        raise TimeSeriesSplitError("Test dates contain duplicates")

    if split.train_end >= split.test_start:
        raise TimeSeriesSplitError("Train dates must end before test dates start")

    overlap = split.train_dates.intersection(split.test_dates)

    if not overlap.empty:
        raise TimeSeriesSplitError("Train and test dates cannot overlap")


def _prepare_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(dates, pd.DatetimeIndex):
        raise TimeSeriesSplitError("Dates must be provided as a DatetimeIndex")

    if dates.empty:
        raise TimeSeriesSplitError("Dates cannot be empty")

    if dates.has_duplicates:
        raise TimeSeriesSplitError("Dates contain duplicate values")

    return pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()


def _validate_split_parameters(
    train_size: int,
    test_size: int,
    step_size: int | None,
    max_splits: int | None,
) -> int:
    if train_size <= 0:
        raise TimeSeriesSplitError("Train size must be positive")

    if test_size <= 0:
        raise TimeSeriesSplitError("Test size must be positive")

    if step_size is None:
        clean_step_size = test_size
    else:
        clean_step_size = step_size

    if clean_step_size <= 0:
        raise TimeSeriesSplitError("Step size must be positive")

    if max_splits is not None and max_splits <= 0:
        raise TimeSeriesSplitError("Maximum splits must be positive")

    return clean_step_size


def _build_split(
    split_id: int,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> TimeSeriesSplit:
    split = TimeSeriesSplit(
        split_id=split_id,
        train_start=pd.Timestamp(train_dates[0]),
        train_end=pd.Timestamp(train_dates[-1]),
        test_start=pd.Timestamp(test_dates[0]),
        test_end=pd.Timestamp(test_dates[-1]),
        train_dates=train_dates,
        test_dates=test_dates,
    )

    validate_time_series_split(split)

    return split
