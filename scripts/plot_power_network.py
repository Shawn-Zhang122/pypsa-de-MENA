# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates plots for optimised power network topologies and regional generation,
storage and conversion capacities built.

Cleaned for pandas>=2 / PyPSA>=1.x robustness:
- NEVER pass pandas Series/StringDtype as colors into n.plot() or legends
- Convert all color mappings to plain Python dict[str, str] / list[str]
- Keep bus_sizes as numeric Series (OK), but force its index to plain tuples
- Provide n.plot() keyword compatibility (PyPSA uses either singular or plural kwargs)
"""

import logging

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

from scripts._helpers import configure_logging, rename_techs, retry, set_scenario_config
from scripts.make_summary import assign_locations
from scripts.plot_summary import preferred_order

logger = logging.getLogger(__name__)


# ------------------------------
# pandas>=2 safety helpers
# ------------------------------
def _disable_pandas_string_inference() -> None:
    # Avoid Arrow/StringDtype surprises in pandas>=2.1
    try:
        pd.set_option("future.infer_string", False)
    except Exception:
        pass


def _safe_str(x, default: str = "lightgrey") -> str:
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    s = str(x)
    return default if s.strip() == "" else s


def _as_color_dict(d, default: str = "lightgrey") -> dict[str, str]:
    """
    Convert (possibly) pandas objects into a plain dict[str, str] of colors.
    """
    out: dict[str, str] = {}
    if d is None:
        return out
    if isinstance(d, dict):
        for k, v in d.items():
            out[str(k)] = _safe_str(v, default=default)
        return out
    if isinstance(d, (pd.Series, pd.Index)):
        for k, v in d.items():
            out[str(k)] = _safe_str(v, default=default)
        return out
    # fallback: try iter of pairs
    try:
        for k, v in d:
            out[str(k)] = _safe_str(v, default=default)
    except Exception:
        pass
    return out


def _call_n_plot_compat(n: pypsa.Network, ax, **kwargs):
    """
    PyPSA plot API differs across versions: some use plural (bus_colors),
    others use singular (bus_color). We try both deterministically.
    """
    # First try as-is
    try:
        return n.plot(ax=ax, **kwargs)
    except TypeError as e1:
        # Map plural <-> singular for key args
        kw = dict(kwargs)

        plural_to_singular = {
            "bus_colors": "bus_color",
            "line_colors": "line_color",
            "link_colors": "link_color",
            "bus_sizes": "bus_size",
            "line_widths": "line_width",
            "link_widths": "link_width",
        }
        singular_to_plural = {v: k for k, v in plural_to_singular.items()}

        # If user passed plural, convert to singular; else convert singular to plural
        changed = False
        for k, v in list(kw.items()):
            if k in plural_to_singular:
                kw[plural_to_singular[k]] = v
                kw.pop(k)
                changed = True
            elif k in singular_to_plural:
                kw[singular_to_plural[k]] = v
                kw.pop(k)
                changed = True

        if not changed:
            raise e1

        try:
            return n.plot(ax=ax, **kw)
        except TypeError:
            # Re-raise the original, more informative error
            raise e1


# ------------------------------
# domain helpers
# ------------------------------
def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["NH3", "Haber-Bosch", "ammonia cracker", "ammonia store"]:
        return "ammonia"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def _sanitize_network_for_plot(n: pypsa.Network) -> None:
    """
    Plot-only sanitisation:
    - ensure n.carriers.color is plain python strings (no StringDtype / ArrowStringArray)
    - ensure buses carrier isn't empty
    """
    if "color" in n.carriers.columns:
        n.carriers["color"] = (
            n.carriers["color"].astype(object).replace("", "lightgrey").fillna("lightgrey")
        )

    # Remove empty carrier name entirely (avoids ambiguous lookups)
    if "" in n.carriers.index:
        n.carriers = n.carriers[n.carriers.index != ""]

    if "carrier" in n.buses.columns:
        n.buses["carrier"] = (
            n.buses["carrier"].astype(object).replace("", "unknown").fillna("unknown")
        )


def _build_tech_colors_from_config(plotting_params) -> dict[str, str]:
    raw = plotting_params.get("tech_colors", {})
    tech_colors = _as_color_dict(raw, default="lightgrey")

    # Ensure placeholder carriers exist
    for k in ["", "none", "unknown"]:
        tech_colors.setdefault(k, "lightgrey")

    # Normalize empties
    for k, v in list(tech_colors.items()):
        tech_colors[k] = _safe_str(v, default="lightgrey")

    return tech_colors


# ------------------------------
# main plot
# ------------------------------
@retry
def plot_map(
    n,
    components=("links", "stores", "storage_units", "generators"),
    bus_size_factor=2e10,
    transmission=False,
    with_legend=True,
):
    _sanitize_network_for_plot(n)

    tech_colors = _build_tech_colors_from_config(snakemake.params.plotting)

    assign_locations(n)

    # Drop non-electric buses so they don't clutter the plot
    if "carrier" in n.buses.columns:
        n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)
        if df_c.empty:
            continue

        df_c = df_c.copy()
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = (
            (df_c.capital_cost * df_c[attr])
            .groupby([df_c.location, df_c.nice_group])
            .sum()
            .unstack()
            .fillna(0.0)
        )
        costs = pd.concat([costs, costs_c], axis=1)

        logger.debug(f"{comp}, {costs}")

    # merge duplicated columns (same tech name may appear across components)
    costs = costs.T.groupby(costs.columns).sum().T

    # remove zero columns
    costs.drop(list(costs.columns[(costs == 0.0).all()]), axis=1, inplace=True)

    new_columns = preferred_order.intersection(costs.columns).append(
        costs.columns.difference(preferred_order)
    )
    costs = costs[new_columns]

    for item in new_columns:
        if item not in tech_colors:
            logger.warning(f"{item} not in config/plotting/tech_colors")

    # stack -> MultiIndex (bus, carrier); values numeric
    costs = costs.stack()

    # hack because impossible to drop buses...
    eu_location = snakemake.params.plotting.get("eu_node_location", dict(x=-5.5, y=46))
    if "EU gas" in n.buses.index:
        n.buses.loc["EU gas", "x"] = eu_location["x"]
        n.buses.loc["EU gas", "y"] = eu_location["y"]

    # only keep DC/B2B links for plotting
    if not n.links.empty and "carrier" in n.links.columns:
        n.links.drop(
            n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")],
            inplace=True,
        )

    # drop non-bus entries from costs
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        logger.info(f"Dropping non-buses {to_drop.tolist()}")
        costs = costs.drop(to_drop, level=0, axis=0, errors="ignore")

    # Ensure the MultiIndex contains plain python tuples (avoid pandas extension dtypes)
    costs.index = pd.MultiIndex.from_tuples([(str(a), str(b)) for a, b in costs.index])

    # filter carriers shown in legend by threshold
    threshold = 100e6  # 100 mEUR/a
    carriers_sum = costs.groupby(level=1).sum()
    carriers_sum = carriers_sum.where(carriers_sum > threshold).dropna()
    carriers = list(map(str, carriers_sum.index))

    # Line/link widths
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 4e3
    ac_color = "rosybrown"
    dc_color = "darkseagreen"
    title = "added grid"

    if snakemake.params.transmission_limit == "lv1.0":
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.0
            title = "current grid"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "total grid"

    line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold).replace(
        line_lower_threshold, 0
    )
    link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold).replace(
        line_lower_threshold, 0
    )

    # ---- plot ----
    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    # IMPORTANT:
    # - bus_colors must be plain dict[str,str] (NOT Series)
    # - legend colors must be list[str] (NOT Series)
    _call_n_plot_compat(
        n,
        ax=ax,
        bus_sizes=costs / float(bus_size_factor),
        bus_colors=dict(tech_colors),
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=(line_widths.to_numpy(dtype=float, copy=False) / float(linewidth_factor)),
        link_widths=(link_widths.to_numpy(dtype=float, copy=False) / float(linewidth_factor)),
        **map_opts,
    )

    # ---- legend: circles ----
    sizes = [20, 10, 5]
    labels = [f"{s} bEUR/a" for s in sizes]
    sizes = [s / float(bus_size_factor) * 1e9 for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.06),
        labelspacing=0.8,
        frameon=False,
        handletextpad=0,
        title="system cost",
    )
    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    # ---- legend: lines ----
    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / float(linewidth_factor)
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.27, 1.06),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
        title=title,
    )
    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="lightgrey"), legend_kw=legend_kw
    )

    # ---- legend: patches (carriers) ----
    legend_kw = dict(
        bbox_to_anchor=(1.52, 1.04),
        frameon=False,
    )
    if with_legend:
        colors = [tech_colors.get(c, "lightgrey") for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]
        add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    fig.savefig(snakemake.output.map, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _disable_pandas_string_inference()

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_power_network",
            opts="",
            clusters="37",
            sector_opts="4380H-T-H-B-I-A-dist1",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    _sanitize_network_for_plot(n)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = snakemake.params.plotting["map"]

    if map_opts.get("boundaries") is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    proj = load_projection(snakemake.params.plotting)

    plot_map(n)