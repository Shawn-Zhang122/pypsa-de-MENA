# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
Plot clustered power network (PyPSA v1.x + pandas>=2 robust).
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from _helpers import set_scenario_config
from scripts.plot_power_network import load_projection


def _disable_pandas_string_inference() -> None:
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
            car = _safe_str(n.buses.at[b, "carrier"], default="unknown")
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
        snakemake = mock_snakemake("plot_power_network_clustered", run="ExPol", clusters=27)

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

    if not n.lines.empty:
        line_width = (n.lines["s_nom"].to_numpy(dtype=float, copy=False) / float(lw_factor))
    else:
        line_width = 1.5

    if not n.links.empty:
        link_width = (n.links["p_nom"].to_numpy(dtype=float, copy=False) / float(lw_factor))
    else:
        link_width = 1.5

    n.plot(
        ax=ax,
        margin=0.06,
        bus_color=bus_color,
        line_color="lightsteelblue",
        link_color=link_color,
        line_width=line_width,
        link_width=link_width,
    )

    plt.savefig(snakemake.output.map, bbox_inches="tight")
    plt.close()