# plot_base_network.py
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT

"""
Plot base transmission network (PyPSA v1.x + pandas>=2 robust).

Fix:
- Prevent pandas StringDtype from reaching numpy in apply_cmap
- Provide already-final colors (strings), not values that require cmap
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from _helpers import set_scenario_config
from scripts.plot_power_network import load_projection
from pypsa.plot import add_legend_lines


def _disable_pandas_string_inference() -> None:
    """
    pandas>=2.1 may infer StringDtype via the future option.
    If enabled, PyPSA/pandas can end up with StringDtype and numpy fails in issubdtype().
    """
    try:
        pd.set_option("future.infer_string", False)
    except Exception:
        pass


def _safe_str(x, default: str = "lightgrey") -> str:
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    x = str(x)
    return default if x.strip() == "" else x


def _carrier_color_map(n: pypsa.Network, default: str = "lightgrey") -> dict[str, str]:
    cmap: dict[str, str] = {}
    if "color" in n.carriers.columns:
        for c in n.carriers.index:
            cmap[str(c)] = _safe_str(n.carriers.at[c, "color"], default=default)
    else:
        for c in n.carriers.index:
            cmap[str(c)] = default
    cmap.setdefault("unknown", default)
    cmap.setdefault("", default)
    return cmap


def _bus_color_dict(n: pypsa.Network, default: str = "lightgrey") -> dict[str, str]:
    cmap = _carrier_color_map(n, default=default)
    out: dict[str, str] = {}

    if "carrier" in n.buses.columns:
        for b in n.buses.index:
            car = n.buses.at[b, "carrier"]
            car = _safe_str(car, default="unknown")
            out[str(b)] = cmap.get(str(car), default)
    else:
        for b in n.buses.index:
            out[str(b)] = default

    return out


def _link_color_dict(n: pypsa.Network, default: str = "lightgrey") -> dict[str, str]:
    out: dict[str, str] = {}
    if n.links.empty:
        return out

    p_nom = n.links["p_nom"].to_numpy(dtype=float, copy=False)
    for i, name in enumerate(map(str, n.links.index)):
        val = p_nom[i]
        if np.isnan(val):
            out[name] = default
        else:
            out[name] = "darkseagreen" if val > 0 else "skyblue"
    return out


if __name__ == "__main__":
    _disable_pandas_string_inference()

    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_base_network", run="tyndp")

    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    lw_factor = 1e3 if n.lines.empty else 2e3

    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    proj = load_projection(snakemake.params.plotting)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj})
    regions.to_crs(proj.proj4_init).plot(
        ax=ax, facecolor="none", edgecolor="lightgray", linewidth=0.75
    )

    bus_color = _bus_color_dict(n, default="lightgrey")
    link_color = _link_color_dict(n, default="lightgrey")

    # Widths: force plain float
    if not n.lines.empty:
        line_width = n.lines["s_nom"].to_numpy(dtype=float, copy=False) / float(lw_factor)
    else:
        line_width = 1.5

    if not n.links.empty:
        link_width = n.links["p_nom"].to_numpy(dtype=float, copy=False) / float(lw_factor)
    else:
        link_width = 1.5

    # PyPSA v1.x plotting keywords (singular)
    n.plot(
        ax=ax,
        margin=0.06,
        bus_color=bus_color,
        line_color="lightsteelblue",
        link_color=link_color,
        line_width=line_width,
        link_width=link_width,
    )

    if not n.lines.empty:
        sizes_ac = [10, 20]
        labels_ac = [f"HVAC ({s} GW)" for s in sizes_ac]
        scale_ac = 1e3 / lw_factor
        sizes_ac = [s * scale_ac for s in sizes_ac]

        add_legend_lines(
            ax,
            sizes_ac,
            labels_ac,
            patch_kw=dict(color="rosybrown"),
            legend_kw=dict(
                loc=[0.25, 0.9],
                frameon=False,
                labelspacing=0.5,
                handletextpad=1,
                fontsize=13,
            ),
        )

    if not n.links.empty:
        sizes_dc = [1, 5]
        labels_dc = [f"HVDC ({s} GW)" for s in sizes_dc]
        scale_dc = 1e3 / lw_factor
        sizes_dc = [s * scale_dc for s in sizes_dc]

        add_legend_lines(
            ax,
            sizes_dc,
            labels_dc,
            patch_kw=dict(color="darkseagreen"),
            legend_kw=dict(
                loc=[0.0, 0.9],
                frameon=False,
                labelspacing=0.5,
                handletextpad=1,
                fontsize=13,
            ),
        )

    plt.savefig(snakemake.output.map, bbox_inches="tight")
    plt.close()