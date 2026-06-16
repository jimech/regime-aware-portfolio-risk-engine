import pandas as pd
import pytest

from regime_risk_engine.validation.splits import (
    TimeSeriesSplit,
    TimeSeriesSplitError,
    build_expanding_window_splits,
    build_rolling_window_splits,
    split_frame_by_dates,
    validate_time_series_split,
)


def make_dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=10, freq="D")


def test_build_expanding_window_splits() -> None:
    splits = build_expanding_window_splits(
        dates=make_dates(),
        initial_train_size=4,
        test_size=2,
    )

    assert len(splits) == 3

    first_split = splits[0]
    second_split = splits[1]
    third_split = splits[2]

    assert first_split.split_id == 0
    assert len(first_split.train_dates) == 4
    assert len(first_split.test_dates) == 2
    assert first_split.train_start == pd.Timestamp("2020-01-01")
    assert first_split.train_end == pd.Timestamp("2020-01-04")
    assert first_split.test_start == pd.Timestamp("2020-01-05")
    assert first_split.test_end == pd.Timestamp("2020-01-06")

    assert len(second_split.train_dates) == 6
    assert second_split.test_start == pd.Timestamp("2020-01-07")

    assert len(third_split.train_dates) == 8
    assert third_split.test_start == pd.Timestamp("2020-01-09")


def test_build_rolling_window_splits() -> None:
    splits = build_rolling_window_splits(
        dates=make_dates(),
        train_size=4,
        test_size=2,
    )

    assert len(splits) == 3

    first_split = splits[0]
    second_split = splits[1]
    third_split = splits[2]

    assert len(first_split.train_dates) == 4
    assert first_split.train_start == pd.Timestamp("2020-01-01")
    assert first_split.train_end == pd.Timestamp("2020-01-04")
    assert first_split.test_start == pd.Timestamp("2020-01-05")

    assert len(second_split.train_dates) == 4
    assert second_split.train_start == pd.Timestamp("2020-01-03")
    assert second_split.test_start == pd.Timestamp("2020-01-07")

    assert len(third_split.train_dates) == 4
    assert third_split.train_start == pd.Timestamp("2020-01-05")
    assert third_split.test_start == pd.Timestamp("2020-01-09")


def test_build_splits_with_custom_step_size() -> None:
    splits = build_rolling_window_splits(
        dates=make_dates(),
        train_size=4,
        test_size=2,
        step_size=1,
    )

    assert len(splits) == 5
    assert splits[0].test_start == pd.Timestamp("2020-01-05")
    assert splits[1].test_start == pd.Timestamp("2020-01-06")


def test_build_splits_with_max_splits() -> None:
    splits = build_expanding_window_splits(
        dates=make_dates(),
        initial_train_size=4,
        test_size=2,
        max_splits=2,
    )

    assert len(splits) == 2


def test_split_frame_by_dates_with_datetime_index() -> None:
    data = pd.DataFrame(
        {"value": range(10)},
        index=make_dates(),
    )
    split = build_expanding_window_splits(
        dates=make_dates(),
        initial_train_size=4,
        test_size=2,
        max_splits=1,
    )[0]

    train_frame, test_frame = split_frame_by_dates(data, split)

    assert len(train_frame) == 4
    assert len(test_frame) == 2
    assert train_frame.index.max() == pd.Timestamp("2020-01-04")
    assert test_frame.index.min() == pd.Timestamp("2020-01-05")


def test_split_frame_by_dates_with_date_column() -> None:
    data = pd.DataFrame(
        {
            "date": make_dates(),
            "value": range(10),
        }
    )
    split = build_rolling_window_splits(
        dates=make_dates(),
        train_size=4,
        test_size=2,
        max_splits=1,
    )[0]

    train_frame, test_frame = split_frame_by_dates(
        data=data,
        split=split,
        date_col="date",
    )

    assert len(train_frame) == 4
    assert len(test_frame) == 2
    assert train_frame["date"].max() == pd.Timestamp("2020-01-04")
    assert test_frame["date"].min() == pd.Timestamp("2020-01-05")


def test_validate_time_series_split_success() -> None:
    split = build_expanding_window_splits(
        dates=make_dates(),
        initial_train_size=4,
        test_size=2,
        max_splits=1,
    )[0]

    validate_time_series_split(split)


def test_validate_time_series_split_rejects_overlap() -> None:
    split = TimeSeriesSplit(
        split_id=0,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-03"),
        test_start=pd.Timestamp("2020-01-03"),
        test_end=pd.Timestamp("2020-01-04"),
        train_dates=pd.date_range("2020-01-01", periods=3, freq="D"),
        test_dates=pd.date_range("2020-01-03", periods=2, freq="D"),
    )

    with pytest.raises(TimeSeriesSplitError, match="before test"):
        validate_time_series_split(split)


def test_build_splits_rejects_non_datetime_index() -> None:
    dates = pd.Index([0, 1, 2])

    with pytest.raises(TimeSeriesSplitError, match="DatetimeIndex"):
        build_expanding_window_splits(
            dates=dates,  # type: ignore[arg-type]
            initial_train_size=2,
            test_size=1,
        )


def test_build_splits_rejects_duplicate_dates() -> None:
    dates = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
        ]
    )

    with pytest.raises(TimeSeriesSplitError, match="duplicate"):
        build_expanding_window_splits(
            dates=dates,
            initial_train_size=2,
            test_size=1,
        )


def test_build_splits_rejects_invalid_train_size() -> None:
    with pytest.raises(TimeSeriesSplitError, match="Train size"):
        build_rolling_window_splits(
            dates=make_dates(),
            train_size=0,
            test_size=2,
        )


def test_build_splits_rejects_invalid_test_size() -> None:
    with pytest.raises(TimeSeriesSplitError, match="Test size"):
        build_rolling_window_splits(
            dates=make_dates(),
            train_size=4,
            test_size=0,
        )


def test_build_splits_rejects_not_enough_dates() -> None:
    with pytest.raises(TimeSeriesSplitError, match="Not enough dates"):
        build_expanding_window_splits(
            dates=pd.date_range("2020-01-01", periods=3, freq="D"),
            initial_train_size=3,
            test_size=2,
        )


def test_split_frame_by_dates_rejects_non_datetime_index() -> None:
    data = pd.DataFrame(
        {"value": [1, 2]},
        index=[0, 1],
    )
    split = build_expanding_window_splits(
        dates=make_dates(),
        initial_train_size=4,
        test_size=2,
        max_splits=1,
    )[0]

    with pytest.raises(TimeSeriesSplitError, match="DatetimeIndex"):
        split_frame_by_dates(data, split)


def test_split_frame_by_dates_rejects_missing_date_column() -> None:
    data = pd.DataFrame({"value": [1, 2]})
    split = build_expanding_window_splits(
        dates=make_dates(),
        initial_train_size=4,
        test_size=2,
        max_splits=1,
    )[0]

    with pytest.raises(TimeSeriesSplitError, match="Date column"):
        split_frame_by_dates(
            data=data,
            split=split,
            date_col="date",
        )
