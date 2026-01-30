# plot_balance_map.py
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create energy balance maps for the defined carriers.

Patch (plot-only; no .nc changes):
1) Fix PyPSA statistics unit crash:
   AttributeError: 'ArrowStringArray' object has no attribute 'item'
   by overriding n.bus_carrier_unit(...) to return a safe scalar string.

2) Fix PyPSA plotting crash:
   TypeError: Cannot interpret '<StringDtype(...)>' as a data type
   by disabling pandas StringDtype inference and ensuring colors passed into n.plot are
   plain Python strings/dicts (not pandas StringDtype Series).
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import sanitize_carriers
from scripts.plot_power_network import load_projection

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1


def _disable_pandas_string_inference() -> None:
    try:
        pd.set_option("future.infer_string", False)
    except Exception:
        pass


def _safe_unit_scalar(x, default="MWh") -> str:
    """
    Accept scalars / pandas arrays / ArrowStringArray and return a scalar unit string.
    Preference: first non-empty string, else default.
    """
    if x is None:
        return default

    # pandas/arrow arrays or list-like
    if hasattr(x, "to_list"):
        try:
            vals = x.to_list()
        except Exception:
            vals = list(x)
        for v in vals:
            if v is None:
                continue
            s = str(v).strip()
            if s != "" and s.lower() != "nan":
                return s
        return default

    s = str(x).strip()
    return default if s == "" or s.lower() == "nan" else s


def _patch_bus_carrier_unit(n: pypsa.Network, default="MWh") -> None:
    """
    Override n.bus_carrier_unit(bus_carrier) to avoid ArrowStringArray.item() crash.
    """
    import types

    def _bus_carrier_unit(self, bus_carrier):
        # best-effort: read from carriers['unit'] if present
        if hasattr(self, "carriers") and "unit" in self.carriers.columns:
            if bus_carrier in self.carriers.index:
                u = self.carriers.loc[bus_carrier, "unit"]
                return _safe_unit_scalar(u, default=default)
        return default

    n.bus_carrier_unit = types.MethodType(_bus_carrier_unit, n)


def _ensure_carrier_columns_safe(n: pypsa.Network) -> None:
    # colors: keep plain python strings
    if "color" in n.carriers.columns:
        n.carriers["color"] = (
            n.carriers["color"]
            .astype(object)
            .replace(r"^\s*$", np.nan, regex=True)
            .fillna("lightgrey")
        )
    # units: pick first non-empty per carrier (handles duplicated/mixed units)
    if "unit" in n.carriers.columns:
        # convert to object to avoid ArrowStringArray propagation
        n.carriers["unit"] = (
            n.carriers["unit"]
            .astype(object)
            .replace(r"^\s*$", np.nan, regex=True)
            .fillna(np.nan)
        )
        # if duplicates happen, keep as-is but safe scalar extraction is handled by _patch_bus_carrier_unit


if __name__ == "__main__":
    _disable_pandas_string_inference()

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_balance_map",
            clusters="10",
            opts="",
            sector_opts="",
            planning_horizons="2050",
            carrier="H2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)

    # keep original sanitisation
    sanitize_carriers(n, snakemake.config)

    # ---- hard fixes (plot-only) ----
    _ensure_carrier_columns_safe(n)
    _patch_bus_carrier_unit(n, default="MWh")

    pypsa.options.params.statistics.round = 3
    pypsa.options.params.statistics.drop_zero = True
    pypsa.options.params.statistics.nice_names = False

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    config = snakemake.params.plotting
    carrier = snakemake.wildcards.carrier

    # fill empty colors or "" with light grey (plain object dtype)
    if "color" in n.carriers.columns:
        mask = n.carriers.color.isna() | n.carriers.color.astype(object).eq("")
        n.carriers["color"] = n.carriers.color.astype(object).mask(mask, "lightgrey")

    # set EU location with location from config
    eu_location = config["eu_node_location"]
    if "EU" in n.buses.index:
        n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]

    # get balance map plotting parameters
    boundaries = config["map"]["boundaries"]
    config = config["balance_map"][carrier]
    conversion = config["unit_conversion"]

    if carrier not in n.buses.carrier.unique():
        raise ValueError(
            f"Carrier {carrier} is not in the network. Remove from configuration `plotting: balance_map: bus_carriers`."
        )

    # for plotting change bus to location
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

    # set x and y coordinates to bus location
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)

    # bus_sizes according to energy balance of bus carrier
    eb = n.statistics.energy_balance(bus_carrier=carrier, groupby=["bus", "carrier"])

    # remove energy balance of transmission carriers which relate to losses
    transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
        {"name": "carrier"}
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")

    # only carriers that are also in the energy balance
    carriers_in_eb = carriers[carriers.isin(eb.index.get_level_values("carrier"))]

    eb.loc[components] = eb.loc[components].drop(index=carriers_in_eb, level="carrier")
    eb = eb.dropna()

    bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(conversion)
    bus_sizes = bus_sizes.sort_values(ascending=False)

    # Get colors for carriers
    # n.carriers.update({"color": ...}) can introduce dtype issues; keep plain object strings:
    tech_colors = snakemake.params.plotting.get("tech_colors", {})
    if isinstance(tech_colors, dict) and len(tech_colors) > 0 and "color" in n.carriers.columns:
        for k, v in tech_colors.items():
            if k in n.carriers.index:
                n.carriers.at[k, "color"] = str(v)

    carrier_colors_map = {}
    if "color" in n.carriers.columns:
        for c in n.carriers.index:
            carrier_colors_map[str(c)] = str(n.carriers.at[c, "color"]) if n.carriers.at[c, "color"] else "lightgrey"
    carrier_colors_map.setdefault("", "lightgrey")
    carrier_colors_map.setdefault("unknown", "lightgrey")

    # IMPORTANT: pass a plain dict {carrier: color} (not pandas Series with StringDtype)
    unique_carriers = list(map(str, bus_sizes.index.get_level_values("carrier").unique()))
    colors = {c: carrier_colors_map.get(c, "lightgrey") for c in unique_carriers}

    # line and links widths according to optimal capacity
    flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(conversion)

    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    # if there are no lines or links for the bus carrier, use fallback for plotting
    fallback = pd.Series(dtype=float)
    line_widths = flow.get("Line", fallback).abs()
    link_widths = flow.get("Link", fallback).abs()

    # define maximal size of buses and branch width
    bus_size_factor = config["bus_factor"]
    branch_width_factor = config["branch_factor"]
    flow_size_factor = config["flow_factor"]

    # get prices per region as colormap
    buses = n.buses.query("carrier in @carrier").index
    weights = n.snapshot_weightings.generators
    prices = weights @ n.buses_t.marginal_price[buses] / weights.sum()
    level = "name" if PYPSA_V1 else "Bus"
    price = prices.rename(n.buses.location).groupby(level=level).mean()

    if carrier == "co2 stored" and "CO2Limit" in n.global_constraints.index:
        co2_price = n.global_constraints.loc["CO2Limit", "mu"]
        price = price - co2_price

    # if only one price is available, use this price for all regions
    if price.size == 1:
        regions["price"] = price.values[0]
        shift = round(price.values[0] / 20, 0)
    else:
        regions["price"] = price.reindex(regions.index).fillna(0)
        shift = 0

    vmin, vmax = regions.price.min() - shift, regions.price.max() + shift
    if config["vmin"] is not None:
        vmin = config["vmin"]
    if config["vmax"] is not None:
        vmax = config["vmax"]

    crs = load_projection(snakemake.params.plotting)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

    line_flow = flow.get("Line")
    link_flow = flow.get("Link")
    transformer_flow = flow.get("Transformer")

    n.plot(
        bus_sizes=bus_sizes * bus_size_factor,
        bus_colors=colors,  # dict: carrier -> color (safe)
        bus_split_circles=True,
        line_widths=line_widths * branch_width_factor,
        link_widths=link_widths * branch_width_factor,
        line_flow=line_flow * flow_size_factor if line_flow is not None else None,
        link_flow=link_flow * flow_size_factor if link_flow is not None else None,
        transformer_flow=transformer_flow * flow_size_factor
        if transformer_flow is not None
        else None,
        ax=ax,
        margin=0.2,
        geomap=True,
        boundaries=boundaries,
    )

    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="price",
        cmap=config["cmap"],
        vmin=vmin,
        vmax=vmax,
        edgecolor="None",
        linewidth=0,
    )

    ax.set_title(carrier)

    # Add colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=config["cmap"], norm=norm)
    price_unit = config["region_unit"]
    cbr = fig.colorbar(
        sm,
        ax=ax,
        label=f"Average Marginal Price [{price_unit}]",
        shrink=0.95,
        pad=0.03,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")

    # add legend
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }

    pad = 0.18

    # Get lists for supply and consumption carriers
    pos_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
    neg_carriers = bus_sizes[bus_sizes < 0].index.unique("carrier")

    # Determine larger total absolute value for supply and consumption for a carrier
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(_carrier, sign):
        values = bus_sizes.loc[:, _carrier]
        return values[values * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    # Add supply carriers
    add_legend_patches(
        ax,
        [colors.get(str(c), "lightgrey") for c in supp_carriers],
        [str(c) for c in supp_carriers],
        legend_kw={
            "bbox_to_anchor": (0, -pad),
            "ncol": 1,
            "title": "Supply",
            **legend_kwargs,
        },
    )

    # Add consumption carriers
    add_legend_patches(
        ax,
        [colors.get(str(c), "lightgrey") for c in cons_carriers],
        [str(c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    # Add bus legend
    legend_bus_sizes = config["bus_sizes"]
    carrier_unit = config["unit"]
    if legend_bus_sizes is not None:
        add_legend_semicircles(
            ax,
            [
                s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_sizes
            ],
            [f"{s} {carrier_unit}" for s in legend_bus_sizes],
            patch_kw={"color": "#666"},
            legend_kw={
                "bbox_to_anchor": (0, 1),
                **legend_kwargs,
            },
        )

    # Add branch legend
    legend_branch_sizes = config["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {carrier_unit}" for s in legend_branch_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
        )

    fig.savefig(
        snakemake.output[0],
        dpi=400,
        bbox_inches="tight",
    )