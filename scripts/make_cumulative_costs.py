import logging
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

idx = pd.IndexSlice
logger = logging.getLogger(__name__)


# =========================
# 1) Robust reader (supports PyPSA meta+2line header)
# =========================
def _read_costs_csv_strict(path: str) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns:
      df_wide: columns = ['cost_type','component','carrier', <year columns...>]
      years:   list[int] planning horizons found in file (sorted)
    Supports two formats:

    Format A (PyPSA export in your uploaded file):
        cluster,,,27,27,27
        opt,,,none,none,none
        planning_horizon,,,2020,2025,2050
        cost,component,carrier,,,
        capital,Generator,onwind,....

      Here, the year labels are in the 'planning_horizon' row, while the header row
      has blank horizon names. We must stitch them together deterministically.

    Format B (normal wide):
        cost_type,component,carrier,2020,2025,2050
        capital,Generator,onwind,...
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)

    if raw.shape[1] < 4:
        raise ValueError(f"costs.csv has too few columns ({raw.shape[1]}). Need >=4.")

    # ---- Detect "planning_horizon" meta row (Format A) ----
    first_col = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    ph_rows = raw.index[first_col == "planning_horizon"].tolist()

    if ph_rows:
        ph_i = ph_rows[0]
        if ph_i + 1 >= len(raw):
            raise ValueError("Found planning_horizon row but no subsequent header row.")

        # years are in columns from 4th onwards
        years_series = pd.to_numeric(pd.Series(raw.iloc[ph_i, 3:]), errors="coerce")
        years = years_series.dropna().astype(int).tolist()
        years = sorted(years)

        if len(years) < 1:
            raise ValueError(
                "planning_horizon row found but no numeric years detected. "
                "Check costs.csv formatting."
            )

        header_i = ph_i + 1
        header = raw.iloc[header_i, :].tolist()

        # normalize first three header names
        fixed_header = list(header)
        fixed_header[0] = "cost_type"
        fixed_header[1] = "component"
        fixed_header[2] = "carrier"

        # fill horizon header names using the years list, positionally
        # horizon columns begin at index 3
        horizon_slots = raw.shape[1] - 3
        # years list length may be <= horizon_slots; align left
        for k in range(horizon_slots):
            col_j = 3 + k
            h = str(fixed_header[col_j]).strip()
            if h == "" or h.lower() in {"nan", "none"}:
                if k < len(years):
                    fixed_header[col_j] = str(years[k])
                else:
                    # if trailing blank columns exist beyond known years, mark for dropping
                    fixed_header[col_j] = f"__drop_{k}__"
            else:
                # if non-empty, try to parse numeric year; if fails, keep as-is
                try:
                    fixed_header[col_j] = str(int(float(h)))
                except Exception:
                    fixed_header[col_j] = h

        # data begins after header row
        data = raw.iloc[header_i + 1 :, :].copy()
        data.columns = fixed_header

        # drop helper columns
        drop_cols = [c for c in data.columns if str(c).startswith("__drop_")]
        if drop_cols:
            data = data.drop(columns=drop_cols)

        # drop empty rows and rows missing identifiers
        data = data.replace("", np.nan)
        data = data.dropna(how="all")
        data = data.dropna(subset=["cost_type", "component", "carrier"])

        # ensure year columns exist
        year_cols = []
        for c in data.columns[3:]:
            try:
                year_cols.append(int(c))
            except Exception:
                pass

        if not year_cols:
            raise ValueError(
                "Parsed Format A but still found no year columns. "
                "Header stitching failed; please inspect first 10 lines."
            )

        # keep only recognized year columns + ids
        keep_year_cols = sorted(set(year_cols))
        keep_cols = ["cost_type", "component", "carrier"] + [str(y) for y in keep_year_cols]
        data = data.loc[:, keep_cols]

        return data, keep_year_cols

    # ---- Otherwise: assume Format B (normal wide) ----
    df = pd.read_csv(path, dtype=str)
    if df.shape[1] < 4:
        raise ValueError("Normal wide costs.csv must have at least 4 columns.")

    # normalize first three column names
    df = df.rename(
        columns={
            df.columns[0]: "cost_type",
            df.columns[1]: "component",
            df.columns[2]: "carrier",
        }
    )

    # detect year columns from header
    year_cols = []
    for c in df.columns[3:]:
        try:
            year_cols.append(int(float(str(c).strip())))
        except Exception:
            continue

    year_cols = sorted(set(year_cols))
    if not year_cols:
        raise ValueError(
            "No numeric planning horizon columns detected in normal wide header. "
            "Your file likely uses the PyPSA meta+two-line format (Format A)."
        )

    # rename year columns to clean ints-as-strings
    rename_map = {}
    for c in df.columns[3:]:
        s = str(c).strip()
        try:
            rename_map[c] = str(int(float(s)))
        except Exception:
            pass
    if rename_map:
        df = df.rename(columns=rename_map)

    keep_cols = ["cost_type", "component", "carrier"] + [str(y) for y in year_cols]
    df = df.loc[:, keep_cols]

    return df, year_cols


def _wide_to_cost_table(df_wide: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """
    Convert wide df to strict costs table:
      index   = (cost_type, component, carrier)
      columns = Int64Index(years)
      values  = float (annual costs)
    Missing entries are treated as 0.0 (typical for cost components not present in a horizon).
    """
    # melt
    long = df_wide.melt(
        id_vars=["cost_type", "component", "carrier"],
        var_name="planning_horizon",
        value_name="value",
    )

    long["planning_horizon"] = pd.to_numeric(long["planning_horizon"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long.dropna(subset=["planning_horizon"])
    long["planning_horizon"] = long["planning_horizon"].astype(int)

    # keep only known years
    long = long[long["planning_horizon"].isin(years)].copy()

    # build table
    costs = (
        long.set_index(["cost_type", "component", "carrier", "planning_horizon"])["value"]
        .unstack("planning_horizon")
    )

    # ensure complete year columns in declared order
    for y in years:
        if y not in costs.columns:
            costs[y] = np.nan
    costs = costs.loc[:, years]

    # treat missing as 0 (robust + prevents horizon drop)
    costs = costs.fillna(0.0).astype(float)

    # enforce index
    costs.index = pd.MultiIndex.from_tuples(
        costs.index.to_list(),
        names=["cost_type", "component", "carrier"],
    )
    costs.columns = pd.Index([int(y) for y in costs.columns], name="planning_horizon")

    return costs


def _validate_horizons(costs: pd.DataFrame, config_horizons) -> list[int]:
    """
    Enforce contract:
      planning_horizons must be exactly the intersection between data and config,
      in ascending order, and costs is restricted accordingly.
    """
    data_h = [int(c) for c in costs.columns.tolist()]
    data_h = sorted(data_h)

    cfg_h = [int(y) for y in config_horizons]
    cfg_h = sorted(set(cfg_h))

    missing = sorted(set(cfg_h) - set(data_h))
    if missing:
        raise ValueError(
            "Config planning_horizons include years not present in costs.csv.\n"
            f"Missing in data: {missing}\n"
            f"Data horizons: {data_h}\n"
            f"Config horizons: {cfg_h}"
        )

    # Restrict to config years (explicit and validated)
    keep = cfg_h
    costs = costs.loc[:, keep]
    # propagate back (caller holds reference)
    costs.drop(columns=[c for c in costs.columns if c not in keep], inplace=True)
    costs.sort_index(axis=1, inplace=True)

    planning_horizons = [int(c) for c in costs.columns.tolist()]
    planning_horizons = sorted(planning_horizons)

    if planning_horizons != keep:
        raise ValueError(
            "After restriction, costs.columns does not match config horizons.\n"
            f"costs.columns={planning_horizons}\n"
            f"config={keep}"
        )

    return planning_horizons


# =========================
# 2) Core computation (strict invariants)
# =========================
def calculate_cumulative_cost(costs: pd.DataFrame, planning_horizons: list[int]) -> pd.DataFrame:
    # ---- hard invariants ----
    if not isinstance(costs.index, pd.MultiIndex) or costs.index.nlevels != 3:
        raise TypeError("costs.index must be 3-level MultiIndex: (cost_type, component, carrier)")
    if list(costs.columns) != list(planning_horizons):
        raise ValueError(
            "planning_horizons must exactly match costs.columns.\n"
            f"costs.columns={list(costs.columns)}\n"
            f"planning_horizons={planning_horizons}"
        )

    years = np.array(planning_horizons, dtype=int)
    base_year = int(years[0])

    rates = pd.Series(np.arange(0.0, 0.1, 0.01), name="social discount rate")
    out = pd.DataFrame(index=costs.index, columns=rates, dtype=float)

    # (A) discounted sum across horizons for each (cost_type, component, carrier)
    for r in out.columns:
        disc = 1.0 / ((1.0 + float(r)) ** (years - base_year))
        out[r] = (costs.values * disc).sum(axis=1)

    # (B) cumulative cost line: integrate (sum over carriers) along horizon axis (trapz)
    # group by existing (cost_type, component) only
    groups = costs.groupby(level=["cost_type", "component"], sort=False)

    for r in out.columns:
        for (ct, comp), block in groups:
            # block index = carrier
            path = block.loc[:, planning_horizons].sum(axis=0).values.astype(float)

            # NumPy 2.x compatible integration
            integral = float(np.trapezoid(path, x=years))

            out.loc[(ct, comp, "cumulative cost"), r] = integral

    return out


# =========================
# 3) Main
# =========================
if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_cumulative_costs",
            clusters="5",
            opts="",
            sector_opts="",
            configfiles="config/test/config.myopic.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Read strictly (supports your uploaded file format)
    df_wide, data_years = _read_costs_csv_strict(snakemake.input.costs)

    # Convert to strict cost table
    costs = _wide_to_cost_table(df_wide, data_years)

    # Hard sanity check: years must exist
    if costs.shape[1] == 0:
        raise RuntimeError(
            "No planning horizons detected after parsing costs.csv. "
            "This indicates malformed header or missing planning_horizon row."
        )

    # Validate vs config and restrict explicitly
    planning_horizons = _validate_horizons(costs, snakemake.params.scenario["planning_horizons"])

    # Final contract checks
    assert list(costs.columns) == list(planning_horizons)
    assert costs.columns.is_monotonic_increasing

    cumulative_cost = calculate_cumulative_cost(costs, planning_horizons)
    cumulative_cost.to_csv(snakemake.output[0])