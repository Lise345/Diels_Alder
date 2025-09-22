
import re
import io
import ast
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diels–Alder Energy Explorer", layout="wide")

# -----------------------------
# Data loading & cleaning
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    for c in ["G(TS)", "DrG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Normalize list-like columns
    for c in ["diene_subs", "dienophile_subs"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_substituent_field)
    return df

def parse_substituent_field(x):
    """Return a list of substituent SMILES-like strings from the CSV cell."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    # Try Python literal list first
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    # Fallback: extract bracketed tokens like [NH3:1], [CH3:12][OH:13], etc.
    tokens = re.findall(r'\[[^\]]+\](?:\[[^\]]+\])*', s)
    if tokens:
        return tokens
    # Last resort: split on commas and clean
    parts = [p.strip(" []'\"") for p in s.split(",") if p.strip()]
    return parts

# -----------------------------
# Reaction parsing & SMILES helpers
# -----------------------------
def parse_reaction_sides(reaction_smiles: str) -> Tuple[List[str], List[str]]:
    """
    Parse a reaction SMILES 'A.B>>P.Q' into (reactants[], products[]).
    """
    try:
        rxn = str(reaction_smiles)
        if ">>" not in rxn:
            return [p for p in rxn.split(".") if p], []
        left, right = rxn.split(">>", 1)
        reactants = [p for p in left.split(".") if p]
        products = [p for p in right.split(".") if p]
        return reactants, products
    except Exception:
        return [], []

def sanitize_smiles_for_drawer(smiles: str) -> str:
    """Remove atom-mapping numbers like :12 inside brackets so SmilesDrawer renders cleanly."""
    if not smiles:
        return ""
    s = str(smiles)
    s = re.sub(r":\d+", "", s)
    return s

def smilesdrawer_canvas(smiles: str, canvas_id: str, width: int = 160, height: int = 160) -> str:
    s = sanitize_smiles_for_drawer(smiles or "")
    html = f"""
    <div style="border:1px solid #e6e6e6;border-radius:10px;padding:6px;display:inline-block;background:#fff;">
      <canvas id="{canvas_id}" width="{width}" height="{height}"></canvas>
    </div>
    <script>
      (function() {{
        function draw() {{
          var s = {s!r};
          if (!s) return;
          function go() {{
            var options = {{ width: {width}, height: {height}, padding: 6 }};
            var drawer = new SmilesDrawer.Drawer(options);
            SmilesDrawer.parse(s, function(tree) {{
              drawer.draw(tree, "{canvas_id}", "light", false);
            }}, function(err) {{
              var ctx = document.getElementById("{canvas_id}").getContext("2d");
              ctx.font = "12px sans-serif";
              ctx.fillText("Invalid SMILES", 10, 20);
            }});
          }}
          if (window.SmilesDrawer) {{
            go();
          }} else {{
            var sc = document.createElement("script");
            sc.src = "https://cdn.jsdelivr.net/npm/smiles-drawer@2.0.1/dist/smiles-drawer.min.js";
            sc.onload = go;
            document.head.appendChild(sc);
          }}
        }}
        if (document.readyState !== "loading") draw();
        else document.addEventListener("DOMContentLoaded", draw);
      }})();
    </script>
    """
    return html

def caption_for(smi: str) -> str:
    return sanitize_smiles_for_drawer(smi)

# -----------------------------
# Plotting
# -----------------------------
def plot_energy_path(ax, gts: float, drg: float, label: str = "", color=None):
    steps = [0, 1, 2]
    energies = [0.0, float(gts), float(drg)]
    ax.plot(steps, energies, marker='_', markersize=18, markeredgewidth=2, linestyle='dashed', color=color)
    if label:
        ax.text(2.05, energies[-1], label, ha='left', va='center', fontsize=8)

def plot_subset(sub: pd.DataFrame, title: str, mode: str = "auto"):
    if sub.empty:
        st.info("No reactions match your filters.")
        return None
    n = len(sub)
    if mode == "auto":
        mode = "single" if n <= 1 else ("all" if n <= 20 else "hist")

    if mode in ("single", "all"):
        fig, ax = plt.subplots(figsize=(7,5), dpi=120)
        if mode == "single":
            row = sub.iloc[0]
            plot_energy_path(ax, row["G(TS)"], row["DrG"], label=str(row.get("R","")))
        else:
            for _, row in sub.iterrows():
                plot_energy_path(ax, row["G(TS)"], row["DrG"], label=str(row.get("R","")))
        ax.set_xticks([0,1,2], ["Reactants", "TS", "Products"])
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Free energy (kcal/mol)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        st.pyplot(fig, clear_figure=True)
        return fig

    elif mode == "hist":
        fig1, ax1 = plt.subplots(figsize=(7,5), dpi=120)
        ax1.hist(sub["G(TS)"].astype(float).values, bins=30)
        ax1.set_xlabel("G(TS) (kcal/mol)")
        ax1.set_ylabel("Count")
        ax1.set_title(title + " — G(TS) distribution")
        st.pyplot(fig1, clear_figure=True)
        fig2, ax2 = plt.subplots(figsize=(7,5), dpi=120)
        ax2.hist(sub["DrG"].astype(float).values, bins=30)
        ax2.set_xlabel("ΔrG (kcal/mol)")
        ax2.set_ylabel("Count")
        ax2.set_title(title + " — ΔrG distribution")
        st.pyplot(fig2, clear_figure=True)
        return None

# -----------------------------
# Filters
# -----------------------------
def filter_dataframe(df: pd.DataFrame,
                     diene: Optional[str],
                     dienophile: Optional[str],
                     g_range: tuple,
                     r_range: tuple,
                     only_exergonic: bool,
                     only_endergonic: bool) -> pd.DataFrame:
    sub = df.copy()
    if diene:
        sub = sub[sub["diene"] == diene]
    if dienophile:
        sub = sub[sub["dienophile"] == dienophile]
    gmin, gmax = g_range; rmin, rmax = r_range
    sub = sub[(sub["G(TS)"] >= gmin) & (sub["G(TS)"] <= gmax) &
              (sub["DrG"] >= rmin) & (sub["DrG"] <= rmax)]
    if only_exergonic and not only_endergonic:
        sub = sub[sub["DrG"] < 0]
    if only_endergonic and not only_exergonic:
        sub = sub[sub["DrG"] >= 0]
    return sub

# -----------------------------
# UI
# -----------------------------
DATA_PATH = "formatted_dataset.csv"
df = load_data(DATA_PATH)

st.title("🔬 Diels–Alder Energy Explorer")

with st.sidebar:
    st.header("Filters")
    unique_dienes = sorted(df["diene"].dropna().unique().tolist())
    unique_dienophiles = sorted(df["dienophile"].dropna().unique().tolist())

    use_diene = st.checkbox("Filter by diene", value=False)
    diene = st.selectbox("Diene", ["— any —"] + unique_dienes, index=0, disabled=not use_diene)
    diene = None if (not use_diene or diene == "— any —") else diene

    use_dienophile = st.checkbox("Filter by dienophile", value=False)
    dienophile = st.selectbox("Dienophile", ["— any —"] + unique_dienophiles, index=0, disabled=not use_dienophile)
    dienophile = None if (not use_dienophile or dienophile == "— any —") else dienophile

    st.markdown("---")
    gmin_all, gmax_all = float(df["G(TS)"].min()), float(df["G(TS)"].max())
    rmin_all, rmax_all = float(df["DrG"].min()), float(df["DrG"].max())
    g_range = st.slider("G(TS) range (kcal/mol)", min_value=gmin_all, max_value=gmax_all, value=(gmin_all, gmax_all), step=0.1)
    r_range = st.slider("ΔrG range (kcal/mol)", min_value=rmin_all, max_value=rmax_all, value=(rmin_all, rmax_all), step=0.1)

    c1, c2 = st.columns(2)
    only_exergonic = c1.toggle("ΔrG < 0", value=False)
    only_endergonic = c2.toggle("ΔrG ≥ 0", value=False)

    st.markdown("---")
    c3, c4 = st.columns(2)
    sort_by = c3.selectbox("Sort by", ["G(TS)", "DrG", "R", "ID"], index=0)
    topn = int(c4.number_input("Top N (optional)", min_value=0, max_value=1000, value=0, step=1))

    st.markdown("---")
    plot_mode = st.radio("Plot mode", ["Selected only", "All matches", "Auto"], index=0)
    mode_map = {"Selected only": "single", "All matches": "all", "Auto": "auto"}

# Apply filters
subset = filter_dataframe(df, diene, dienophile, g_range, r_range, only_exergonic, only_endergonic)
if sort_by in subset.columns:
    subset = subset.sort_values(sort_by, ascending=True if sort_by in ["G(TS)", "DrG"] else True)
if topn and topn > 0:
    subset = subset.head(topn)

# Header
left_count = f"(n={len(subset)})"
if diene and dienophile:
    st.subheader(f"{diene} + {dienophile}  {left_count}")
elif diene:
    st.subheader(f"{diene}  {left_count}")
elif dienophile:
    st.subheader(f"{dienophile}  {left_count}")
else:
    st.subheader(f"All reactions  {left_count}")

# Quick stats
if not subset.empty:
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Median G(TS)", f"{subset['G(TS)'].median():.2f} kcal/mol")
    with s2:
        st.metric("Median ΔrG", f"{subset['DrG'].median():.2f} kcal/mol")

# Selection of focused reaction row
if not subset.empty:
    labels = [f"{i}: R={row['R']} | G(TS)={row['G(TS)']:.2f} | ΔrG={row['DrG']:.2f}" for i, row in subset.reset_index(drop=True).iterrows()]
    chosen_idx = st.selectbox("Select reaction to focus", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)
    subset = subset.reset_index(drop=True)
    if chosen_idx is not None:
        first = subset.iloc[[chosen_idx]]
        rest = subset.drop(index=chosen_idx)
        subset = pd.concat([first, rest], ignore_index=True)

# Layout: molecules (from reaction SMILES) | plot
colA, colB = st.columns([1,2], gap="large")

with colA:
    st.markdown("#### Molecules from reaction (SMILES)")
    if not subset.empty:
        rxn = str(subset.iloc[0]["reaction"])
        reactants, products = parse_reaction_sides(rxn)
        tabs = st.tabs(["Reactants", "Products"])
        with tabs[0]:
            if reactants:
                n_cols = min(4, max(2, len(reactants)))
                cols = st.columns(n_cols)
                for i, smi in enumerate(reactants):
                    with cols[i % n_cols]:
                        st.components.v1.html(smilesdrawer_canvas(smi, f"canvas_react_{i}", 160, 160), height=180)
                        st.caption(caption_for(smi))
            else:
                st.write("— none —")
        with tabs[1]:
            if products:
                n_cols = min(4, max(2, len(products)))
                cols = st.columns(n_cols)
                for i, smi in enumerate(products):
                    with cols[i % n_cols]:
                        st.components.v1.html(smilesdrawer_canvas(smi, f"canvas_prod_{i}", 160, 160), height=180)
                        st.caption(caption_for(smi))
            else:
                st.write("— none —")
    else:
        st.info("No reaction to display.")

with colB:
    st.markdown("#### Energy path plot")
    if st.button("Plot energy barriers", type="primary", use_container_width=True):
        fig = plot_subset(subset, "Energy profiles", mode=mode_map[plot_mode])
        if fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
            st.download_button("Download plot (PNG)", data=buf.getvalue(), file_name="energy_profiles.png", mime="image/png")

# FULL-WIDTH substituent galleries
st.markdown("---")
st.markdown("### Substituent galleries (selected reaction)")
if not subset.empty:
    row0 = subset.iloc[0]
    dsubs: List[str] = row0.get("diene_subs", []) or []
    psubs: List[str] = row0.get("dienophile_subs", []) or []

    # Normalize: keep as strings, ensure not empty
    dsubs = [str(s) for s in dsubs if str(s).strip()]
    psubs = [str(s) for s in psubs if str(s).strip()]

    # Choose columns to fill the whole width
    def gallery(smiles_list: List[str], base_id: str, title: str):
        st.markdown(f"#### {title} ({len(smiles_list)})")
        if not smiles_list:
            st.write("— none —")
            return
        n = len(smiles_list)
        n_cols = 6 if n >= 12 else (5 if n >= 9 else (4 if n >= 6 else 3))
        cols = st.columns(n_cols, gap="small")
        for i, smi in enumerate(smiles_list):
            with cols[i % n_cols]:
                st.components.v1.html(smilesdrawer_canvas(smi, f"{base_id}_{i}", 130, 130), height=150)
                st.caption(caption_for(smi))

    gallery(dsubs, "dsub", "Diene substituents")
    gallery(psubs, "psub", "Dienophile substituents")
else:
    st.info("No selected reaction; apply filters above.")
    
# Data table & downloads
st.markdown("---")
st.markdown("### Matching reactions")
cols_to_show = [c for c in ["ID", "R", "diene", "diene_subs", "dienophile", "dienophile_subs", "G(TS)", "DrG", "reaction"] if c in subset.columns]
st.dataframe(subset[cols_to_show], use_container_width=True, height=420)
csv_bytes = subset[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_reactions.csv", mime="text/csv")
