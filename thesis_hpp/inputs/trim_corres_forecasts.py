"""Trim the additional CorRES forecasts to the 8,736-hour model horizon.

The script keeps 364 complete days beginning at 00:00 on 2 January. It
validates every selected file before changing anything and stores the original
full-length CSV files in a backup folder.
"""

from pathlib import Path
import shutil

import pandas as pd


WEATHER_YEARS = [
    1990,
    1991,
    1993,
    1994,
    1995,
    1997,
    1998,
    1999,
    2002,
    2005,
    2006,
    2007,
    2009,
    2010,
    2013,
]
FORECAST_NUMBERS = range(1, 6)
EXPECTED_HOURS = 8_736
TIME_COLUMN = "time"
CAPACITY_FACTOR_COLUMN = "Belwind_Pilot"
CREATE_BACKUPS = True

CORRES_DIR = Path(__file__).resolve().parent / "CORRES"
BACKUP_DIR = CORRES_DIR / "_full_length_backups"


def expected_time_index(year: int, timezone) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=1, day=2, hour=0, tz=timezone)
    return pd.date_range(start=start, periods=EXPECTED_HOURS, freq="h")


def validate_and_trim(path: Path, year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing forecast file: {path}")

    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    required_columns = {TIME_COLUMN, CAPACITY_FACTOR_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )

    timestamps = pd.to_datetime(df[TIME_COLUMN], errors="raise")
    if timestamps.duplicated().any():
        duplicates = df.loc[timestamps.duplicated(keep=False), TIME_COLUMN].head(5)
        raise ValueError(
            f"Duplicate timestamps in {path}. Examples: {duplicates.tolist()}"
        )

    timezone = timestamps.dt.tz
    required_times = expected_time_index(year, timezone)

    indexed = df.assign(_parsed_time=timestamps).set_index("_parsed_time")
    missing_times = required_times.difference(indexed.index)
    if not missing_times.empty:
        preview = [str(value) for value in missing_times[:5]]
        raise ValueError(
            f"{path} is missing {len(missing_times)} required hourly timestamps. "
            f"First missing timestamps: {preview}"
        )

    trimmed = indexed.loc[required_times].reset_index(drop=True)

    capacity_factor = pd.to_numeric(
        trimmed[CAPACITY_FACTOR_COLUMN], errors="coerce"
    )
    if capacity_factor.isna().any():
        raise ValueError(f"Non-numeric or missing capacity factors in {path}")
    if not capacity_factor.between(0.0, 1.0).all():
        minimum = capacity_factor.min()
        maximum = capacity_factor.max()
        raise ValueError(
            f"Capacity factors outside [0, 1] in {path}: "
            f"minimum={minimum}, maximum={maximum}"
        )

    if len(trimmed) != EXPECTED_HOURS:
        raise ValueError(
            f"Expected {EXPECTED_HOURS} rows in {path}, obtained {len(trimmed)}"
        )

    return trimmed


def backup_path(path: Path) -> Path:
    relative_path = path.relative_to(CORRES_DIR)
    return BACKUP_DIR / relative_path


def main() -> None:
    prepared_files = []

    # Validate every file before modifying any of them.
    for year in WEATHER_YEARS:
        for forecast in FORECAST_NUMBERS:
            path = CORRES_DIR / f"corres_{year}" / f"P_forecast{forecast}.csv"
            trimmed = validate_and_trim(path, year)
            prepared_files.append((path, trimmed))
            print(
                f"Validated {path.name} for {year}: "
                f"{trimmed.iloc[0][TIME_COLUMN]} to "
                f"{trimmed.iloc[-1][TIME_COLUMN]}"
            )

    print(f"Validated all {len(prepared_files)} forecast files.")

    for path, trimmed in prepared_files:
        if CREATE_BACKUPS:
            destination = backup_path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(path, destination)

        temporary_path = path.with_suffix(".tmp.csv")
        trimmed.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
        print(f"Trimmed {path}")

    print(
        f"Finished: {len(prepared_files)} files now contain "
        f"{EXPECTED_HOURS} hourly rows."
    )
    if CREATE_BACKUPS:
        print(f"Original files are stored in: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
