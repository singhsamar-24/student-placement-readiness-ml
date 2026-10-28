from __future__ import annotations

import json
import textwrap
def _clean_html(raw_html: str) -> str:
    return "\n".join(line.strip() for line in raw_html.strip().splitlines())

from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.clustering import run_clustering
from src.config import CONFIG, MODEL_DIR
from src.data import load_dataset, validate_dataset
from src.eda import dataset_summary
from src.explainability import global_feature_importance, local_contributors
from src.predict import predict_student
from src.readiness import calculate_readiness
from src.recommendations import generate_recommendations, weak_features
from src.train import train_and_persist


st.set_page_config(page_title="Placement Antigravity", page_icon="🚀", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  ·  All CSS lives here — one source of truth
# ─────────────────────────────────────────────────────────────────────────────
STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
/* ── TOKENS ───────────────────────────────────────────────────────────────── */
:root {
    --font:         'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
    --accent:       #1a73e8;
    --accent-rgb:   26, 115, 232;
    --accent-2:     #34a853;
    --accent-3:     #ea4335;
    --accent-4:     #fbbc05;
    --ink:          #0d1117;
    --ink-2:        #1c2030;
    --muted:        #6b7280;
    --muted-2:      #9ca3af;
    --line:         #e5e7eb;
    --line-2:       #f3f4f6;
    --soft:         #f9fafb;
    --panel:        #ffffff;
    --grad-hero:    linear-gradient(135deg, #0d1117 0%, #0f172a 45%, #0c1a2e 100%);
    --shadow-sm:    0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md:    0 4px 16px rgba(0,0,0,.07), 0 2px 6px rgba(0,0,0,.04);
    --shadow-lg:    0 12px 40px rgba(0,0,0,.10), 0 4px 12px rgba(0,0,0,.05);
    --shadow-xl:    0 24px 64px rgba(0,0,0,.12), 0 8px 24px rgba(0,0,0,.06);
    --shadow-glow:  0 0 0 1px rgba(var(--accent-rgb),.12), 0 8px 32px rgba(var(--accent-rgb),.14);
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    16px;
    --radius-xl:    24px;
    --radius-pill:  999px;
    --ease-out:     cubic-bezier(.22, 1, .36, 1);
    --ease-spring:  cubic-bezier(.34, 1.56, .64, 1);
    --dur-fast:     180ms;
    --dur-base:     260ms;
    --dur-slow:     480ms;
    --dur-enter:    700ms;
}

/* ── BASE ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html { scroll-behavior: smooth; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: var(--font) !important;
    background: var(--panel) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Subtle background grid */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(255,255,255,.97), rgba(255,255,255,.97)),
        linear-gradient(90deg, rgba(var(--accent-rgb),.04) 1px, transparent 1px),
        linear-gradient(180deg, rgba(var(--accent-rgb),.035) 1px, transparent 1px) !important;
    background-size: auto, 52px 52px, 52px 52px !important;
}

.block-container {
    max-width: 1280px !important;
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 5rem !important;
    animation: pageEnter var(--dur-enter) var(--ease-out) both;
}

/* Streamlit's inner app view needs overflow visible for full-bleed elements */
[data-testid="stAppViewContainer"] > section {
    overflow: visible !important;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"],
footer,
section[data-testid="stSidebar"],
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

/* Global heading reset */
h1, h2, h3, h4 {
    font-family: var(--font) !important;
    color: var(--ink) !important;
    letter-spacing: -.02em;
    line-height: 1.15;
}

p { line-height: 1.7; }

/* ── NAVBAR ───────────────────────────────────────────────────────────────── */
.top-shell {
    position: sticky;
    top: 0;
    z-index: 100;
    /* Break out of Streamlit block-container max-width to span full viewport */
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 52px;
    padding: 0 2.5rem;
    background: rgba(255,255,255,.97);
    backdrop-filter: blur(20px) saturate(1.6);
    -webkit-backdrop-filter: blur(20px) saturate(1.6);
    border-bottom: 1px solid rgba(229,231,235,.9);
    animation: navDrop 500ms var(--ease-out) both;
    gap: 1rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: .48rem;
    font-weight: 700;
    font-size: .9rem;
    color: var(--ink);
    letter-spacing: -.02em;
    white-space: nowrap;
    text-decoration: none;
    flex-shrink: 0;
}

/* Google-style four-colour mark */
.brand-mark {
    width: 22px;
    height: 22px;
    position: relative;
    flex-shrink: 0;
}
.brand-mark svg { width: 22px; height: 22px; display: block; }

.nav-links {
    display: flex;
    align-items: center;
    gap: 0;
    flex: 1;
    justify-content: center;
}

/* Each top-level nav item — mirrors "Products ↓ Use Cases ↓ Pricing Enterprise Resources ↓" */
.nav-link {
    display: inline-flex;
    align-items: center;
    gap: .22rem;
    padding: .34rem .72rem;
    border-radius: var(--radius-pill);
    font-size: .86rem;
    font-weight: 500;
    color: #374151;
    text-decoration: none;
    letter-spacing: -.01em;
    transition:
        color var(--dur-fast) ease,
        background var(--dur-fast) ease;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
}
.nav-link:hover {
    color: var(--ink);
    background: rgba(0,0,0,.04);
}
.nav-link.active {
    color: var(--ink);
    font-weight: 600;
}
/* Dropdown chevron on items that have sub-pages */
.nav-link .chev {
    font-size: .65rem;
    opacity: .55;
    margin-left: .05rem;
    transition: transform var(--dur-fast) ease;
}
.nav-link:hover .chev { transform: rotate(180deg); }

/* Right side pill — "Download ↓" style black pill */
.nav-badge {
    display: inline-flex;
    align-items: center;
    gap: .38rem;
    padding: .46rem 1rem;
    border-radius: var(--radius-pill);
    background: #111318;
    color: #fff;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: -.01em;
    flex-shrink: 0;
    transition:
        transform var(--dur-base) var(--ease-spring),
        box-shadow var(--dur-base) ease,
        background var(--dur-base) ease;
    cursor: pointer;
}
.nav-badge:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(0,0,0,.2);
    background: #1c2030;
}
.nav-badge .arrow {
    font-size: .75rem;
    opacity: .7;
}
/* remove pulse dot — not in the reference */

/* ── RADIO NAV (Streamlit) ────────────────────────────────────────────────── */
div[data-testid="stRadio"] {
    margin: 0 0 .5rem;
    padding: .5rem 0 .6rem;
    border-bottom: 1px solid var(--line);
}
div[data-testid="stRadio"] > div { gap: .2rem !important; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    font-family: var(--font) !important;
    font-size: .875rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    padding: .36rem .72rem !important;
    border-radius: var(--radius-pill) !important;
    transition:
        color var(--dur-fast) ease,
        background var(--dur-fast) ease,
        transform var(--dur-fast) ease !important;
    cursor: pointer;
}
div[data-testid="stRadio"] label:hover {
    color: var(--ink) !important;
    background: var(--line-2) !important;
    transform: translateY(-1px);
}
div[data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div,
div[data-testid="stRadio"] [aria-checked="true"] {
    color: var(--accent) !important;
    background: rgba(var(--accent-rgb),.08) !important;
}

/* ── HERO  (light, white bg — matches reference screenshot) ──────────────── */
.hero {
    position: relative;
    min-height: 88vh;
    /* Break out of block-container to span full viewport */
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 10vh 2rem 9vh;
    overflow: hidden;
    isolation: isolate;
    background: #ffffff;          /* pure white — matches screenshot */
}

/* The coloured floating particles come from the canvas injected by
   hero_particles_canvas(); no CSS pseudo-elements needed for them.
   We keep ::after only for an ultra-subtle radial vignette at the edges. */
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 50% 110%, rgba(240,245,255,.9), transparent);
    pointer-events: none;
}

.hero-inner {
    position: relative;
    z-index: 1;
    max-width: 860px;
    margin: 0 auto;
}

/* Small brand badge above the title  — "△ Placement Antigravity" */
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .42rem;
    font-size: .88rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 1.1rem;
    animation: heroEyebrow var(--dur-enter) var(--ease-out) .0s both;
    letter-spacing: -.01em;
}
/* four-colour mini-mark */
.hero-eyebrow .mini-mark {
    width: 16px; height: 16px;
    display: inline-block;
    flex-shrink: 0;
}

/* ── Main title — large, dark, bold, matches "Experience liftoff…" weight ── */
.hero h1 {
    max-width: 840px;
    margin: 0 auto;
    font-size: clamp(3rem, 6.8vw, 7.2rem);
    line-height: .97;
    font-weight: 700;           /* matches reference: not ultra-black, ~700 */
    letter-spacing: -.04em;
    color: var(--ink) !important;
    animation: heroTitle var(--dur-enter) var(--ease-out) .1s both;
}

/* no gradient word in the light version — keep it clean dark like the reference */
.hero h1 .gradient-word {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    background: none !important;
}

/* blinking cursor */
.cursor {
    display: inline-block;
    width: .055em;
    height: .88em;
    margin-left: .06em;
    vertical-align: -.05em;
    border-radius: var(--radius-pill);
    background: var(--muted);
    animation: blink 1.1s steps(1) infinite;
}
@keyframes blink { 50% { opacity: .12; } }

/* copy line hidden in the reference — we keep it but subdued */
.hero-copy {
    display: none;  /* matches the reference: no sub-line visible, just title + CTA */
}

/* ── Single pill CTA — black rounded pill, matches "Download" button ──────── */
.hero-actions {
    display: flex;
    justify-content: center;
    gap: .75rem;
    flex-wrap: wrap;
    margin-top: 2rem;
    animation: heroSub var(--dur-enter) var(--ease-out) .28s both;
}

.hero-cta {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: .42rem;
    min-height: 48px;
    padding: .76rem 1.6rem;
    border-radius: var(--radius-pill);
    font-family: var(--font);
    font-size: .92rem;
    font-weight: 600;
    text-decoration: none;
    transition:
        transform var(--dur-base) var(--ease-spring),
        box-shadow var(--dur-base) ease,
        background var(--dur-base) ease;
    cursor: pointer;
    border: none;
    outline: none;
    -webkit-font-smoothing: antialiased;
}
/* primary = solid black pill (matches "Download" in reference) */
.hero-cta.primary {
    background: #111318;
    color: #fff;
    box-shadow: 0 2px 12px rgba(0,0,0,.16);
}
.hero-cta.primary:hover {
    transform: translateY(-2px) scale(1.015);
    box-shadow: 0 8px 24px rgba(0,0,0,.22);
    background: #1c2030;
}
/* secondary = ghost, subtle */
.hero-cta.secondary {
    background: transparent;
    border: 1.5px solid rgba(0,0,0,.14);
    color: var(--ink);
}
.hero-cta.secondary:hover {
    transform: translateY(-2px) scale(1.015);
    background: var(--line-2);
    border-color: rgba(0,0,0,.22);
}
.hero-cta:active {
    transform: translateY(0) scale(.97) !important;
    transition-duration: 80ms !important;
}

/* ── Stat strip — kept but adapted to light theme ─────────────────────────── */
.hero-stats {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2.5rem;
    margin-top: 3.5rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(0,0,0,.06);
    animation: heroSub var(--dur-enter) var(--ease-out) .44s both;
    flex-wrap: wrap;
}
.hero-stat { text-align: center; }
.hero-stat-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -.04em;
    line-height: 1;
}
.hero-stat-label {
    font-size: .72rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: .05em;
    text-transform: uppercase;
    margin-top: .32rem;
}
.hero-stat-sep {
    width: 1px;
    height: 32px;
    background: rgba(0,0,0,.08);
}

/* ── PAGE SECTION HEADER ──────────────────────────────────────────────────── */
.page-hero {
    padding: 2.8rem 0 1.8rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.8rem;
    animation: heroSub var(--dur-enter) var(--ease-out) .06s both;
}
.page-hero h1 {
    font-size: clamp(2.4rem, 4.8vw, 5rem) !important;
    line-height: 1 !important;
    margin: 0 !important;
    font-weight: 800 !important;
    letter-spacing: -.04em !important;
}
.subtle {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 720px;
    line-height: 1.7;
    margin-top: .6rem;
}

/* ── PILLS & BADGES ───────────────────────────────────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: .32rem;
    padding: .28rem .72rem;
    border-radius: var(--radius-pill);
    background: rgba(var(--accent-rgb),.08);
    color: var(--accent);
    font-weight: 700;
    font-size: .76rem;
    letter-spacing: .04em;
    text-transform: uppercase;
    margin-bottom: .8rem;
}

.section-kicker {
    color: var(--muted);
    font-size: .74rem;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: .5rem;
    display: block;
}

.tag {
    display: inline-flex;
    align-items: center;
    padding: .2rem .56rem;
    border-radius: var(--radius-sm);
    font-size: .74rem;
    font-weight: 600;
    letter-spacing: .02em;
}
.tag-green  { background: #dcfce7; color: #166534; }
.tag-blue   { background: #dbeafe; color: #1e40af; }
.tag-red    { background: #fee2e2; color: #991b1b; }
.tag-amber  { background: #fef3c7; color: #92400e; }

/* ── METRIC CARDS ─────────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1.15rem 1.25rem 1.1rem;
    box-shadow: var(--shadow-sm);
    transition:
        transform var(--dur-base) var(--ease-spring),
        box-shadow var(--dur-base) ease,
        border-color var(--dur-base) ease;
    animation: revealUp var(--dur-enter) var(--ease-out) both;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(var(--accent-rgb),.035) 0%, transparent 60%);
    pointer-events: none;
    opacity: 0;
    transition: opacity var(--dur-base) ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: rgba(var(--accent-rgb),.28);
    box-shadow: var(--shadow-glow);
}
div[data-testid="stMetric"]:hover::before { opacity: 1; }

[data-testid="stMetricLabel"] {
    font-family: var(--font) !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: .04em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: var(--font) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--ink) !important;
    letter-spacing: -.04em;
    line-height: 1.1;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font) !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
}

/* Staggered metric cards */
div[data-testid="stMetric"]:nth-child(1)  { animation-delay:  .04s; }
div[data-testid="stMetric"]:nth-child(2)  { animation-delay:  .10s; }
div[data-testid="stMetric"]:nth-child(3)  { animation-delay:  .16s; }
div[data-testid="stMetric"]:nth-child(4)  { animation-delay:  .22s; }
div[data-testid="stMetric"]:nth-child(5)  { animation-delay:  .28s; }
div[data-testid="stMetric"]:nth-child(6)  { animation-delay:  .34s; }

/* ── CHARTS & DATAFRAMES ──────────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    transition:
        transform var(--dur-base) ease,
        box-shadow var(--dur-base) ease,
        border-color var(--dur-base) ease;
    animation: revealUp var(--dur-enter) var(--ease-out) both;
    background: var(--panel);
}
div[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-3px);
    border-color: rgba(var(--accent-rgb),.2);
    box-shadow: var(--shadow-xl);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    animation: revealUp var(--dur-enter) var(--ease-out) both;
}

/* ── BUTTONS ──────────────────────────────────────────────────────────────── */
.stButton > button,
.stFormSubmitButton > button {
    font-family: var(--font) !important;
    border-radius: var(--radius-pill) !important;
    background: var(--ink) !important;
    color: #fff !important;
    border: 1px solid var(--ink) !important;
    padding: .68rem 1.4rem !important;
    font-size: .9rem !important;
    font-weight: 700 !important;
    letter-spacing: -.01em !important;
    transition:
        transform var(--dur-base) var(--ease-spring),
        box-shadow var(--dur-base) ease,
        background var(--dur-base) ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button::after,
.stFormSubmitButton > button::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(to right, transparent 0%, rgba(255,255,255,.08) 50%, transparent 100%);
    transform: translateX(-100%);
    transition: transform .5s ease;
}
.stButton > button:hover::after,
.stFormSubmitButton > button:hover::after {
    transform: translateX(100%);
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.18) !important;
    background: #1c2030 !important;
    color: #fff !important;
    border-color: #1c2030 !important;
}
.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(0) scale(.97) !important;
    box-shadow: none !important;
    transition-duration: 80ms !important;
}

/* ── FORM INPUTS ──────────────────────────────────────────────────────────── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="textarea"] > div {
    border-radius: var(--radius-md) !important;
    border-color: var(--line) !important;
    transition:
        border-color var(--dur-fast) ease,
        box-shadow var(--dur-fast) ease !important;
    font-family: var(--font) !important;
}
div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div:first-child,
div[data-baseweb="textarea"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(var(--accent-rgb),.12) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(var(--accent-rgb),.2) !important;
    transition: transform var(--dur-fast) var(--ease-spring) !important;
}
[data-testid="stSlider"] > div > div > div[role="slider"]:hover {
    transform: scale(1.18) !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    font-family: var(--font) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color var(--dur-fast) ease, box-shadow var(--dur-fast) ease !important;
}

/* Select box */
div[data-baseweb="select"] {
    transition: box-shadow var(--dur-fast) ease !important;
}

/* ── RESULT CARD ──────────────────────────────────────────────────────────── */
.result-card {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(var(--accent-rgb),.18);
    border-radius: var(--radius-xl);
    padding: clamp(1.5rem, 3vw, 2.2rem);
    background: var(--panel);
    box-shadow: var(--shadow-xl), inset 0 1px 0 rgba(255,255,255,.8);
    animation: resultReveal 700ms var(--ease-out) both;
}
.result-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
        rgba(var(--accent-rgb),.04) 0%,
        transparent 50%,
        rgba(52,168,83,.03) 100%);
    pointer-events: none;
}
.result-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent) 0%, #34a853 50%, #a78bfa 100%);
    border-radius: var(--radius-xl) var(--radius-xl) 0 0;
}

.result-grid {
    display: grid;
    grid-template-columns: minmax(200px, 260px) 1fr;
    gap: clamp(1.2rem, 3vw, 2.5rem);
    align-items: center;
}

/* Circular probability ring */
.prob-ring {
    --p: 75;
    width: min(220px, 55vw);
    aspect-ratio: 1;
    border-radius: 50%;
    display: grid;
    place-items: center;
    margin: 0 auto;
    background:
        radial-gradient(circle closest-side, white 69%, transparent 70%),
        conic-gradient(var(--accent) calc(var(--p) * 1%), #e8eef7 0);
    animation: ringIn 900ms var(--ease-out) both;
    box-shadow:
        inset 0 0 0 1px rgba(0,0,0,.04),
        0 16px 40px rgba(var(--accent-rgb),.15),
        0 4px 12px rgba(0,0,0,.06);
    transition: --p 800ms ease;
}
.prob-ring-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: .12rem;
}
.prob-ring strong {
    font-family: var(--font);
    font-size: clamp(2rem, 4.5vw, 3.8rem);
    line-height: 1;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -.04em;
}
.prob-ring span {
    color: var(--muted);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.result-title {
    margin: 0 0 .5rem;
    font-size: clamp(1.8rem, 3.5vw, 3.5rem);
    line-height: 1;
    font-weight: 800;
    letter-spacing: -.04em;
}
.result-title.placed   { color: #15803d; }
.result-title.risk     { color: #b91c1c; }

.readiness-pill {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    margin-top: 1rem;
    padding: .6rem 1rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-pill);
    background: var(--soft);
    font-weight: 700;
    font-size: .9rem;
    animation: revealUp 600ms var(--ease-out) .2s both;
}

.contributors-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.2rem;
}

.contributor-panel {
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem;
    background: var(--soft);
    animation: revealUp 600ms var(--ease-out) .4s both;
}
.contributor-panel h4 {
    font-size: .82rem !important;
    font-weight: 800 !important;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--muted) !important;
    margin: 0 0 .7rem !important;
}
.contributor-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .38rem 0;
    border-bottom: 1px solid var(--line);
    font-size: .875rem;
    gap: .5rem;
}
.contributor-row:last-child { border-bottom: none; }
.contributor-bar-wrap {
    flex: 1;
    height: 4px;
    background: var(--line);
    border-radius: 99px;
    overflow: hidden;
    margin: 0 .6rem;
}
.contributor-bar {
    height: 100%;
    border-radius: 99px;
    transition: width .8s var(--ease-out);
}
.contributor-bar.positive { background: linear-gradient(90deg, var(--accent), #34a853); }
.contributor-bar.improve  { background: linear-gradient(90deg, #f59e0b, #ef4444); }

.recs-list {
    margin-top: .8rem;
    display: flex;
    flex-direction: column;
    gap: .5rem;
}
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: .7rem;
    padding: .8rem 1rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--panel);
    font-size: .9rem;
    line-height: 1.55;
    animation: revealUp 500ms var(--ease-out) both;
    transition: transform var(--dur-base) ease, box-shadow var(--dur-base) ease;
}
.rec-item:hover {
    transform: translateX(3px);
    box-shadow: var(--shadow-md);
    border-color: rgba(var(--accent-rgb),.2);
}
.rec-item-icon {
    width: 28px; height: 28px;
    border-radius: var(--radius-sm);
    background: rgba(var(--accent-rgb),.08);
    color: var(--accent);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: .9rem;
}

/* ── LOADING STATES ───────────────────────────────────────────────────────── */
.loading-overlay {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
    gap: 1.2rem;
}
.loading-spinner {
    width: 48px; height: 48px;
    border-radius: 50%;
    border: 3px solid var(--line);
    border-top-color: var(--accent);
    animation: spin 700ms linear infinite;
}
.loading-steps {
    display: flex;
    flex-direction: column;
    gap: .45rem;
    min-width: 240px;
}
.loading-step {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-size: .88rem;
    color: var(--muted-2);
    transition: color .3s ease;
}
.loading-step.active  { color: var(--accent); font-weight: 600; }
.loading-step.done    { color: #15803d; }
.loading-step .step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--line);
    flex-shrink: 0;
    transition: background .3s ease, transform .3s var(--ease-spring);
}
.loading-step.active .step-dot {
    background: var(--accent);
    transform: scale(1.4);
    animation: pulse 1s ease infinite;
}
.loading-step.done .step-dot  { background: #34a853; transform: scale(1); }

/* Skeleton shimmer */
.skeleton {
    background: linear-gradient(90deg,
        var(--line-2) 25%,
        var(--line) 50%,
        var(--line-2) 75%);
    background-size: 200% 100%;
    border-radius: var(--radius-md);
    animation: shimmer 1.4s ease infinite;
}
.skeleton-text { height: 14px; margin: .5rem 0; }
.skeleton-card { height: 80px; width: 100%; }

/* ── INFO / ALERT BOXES ───────────────────────────────────────────────────── */
div[data-testid="stInfo"],
div[data-testid="stSuccess"],
div[data-testid="stWarning"],
div[data-testid="stError"] {
    border-radius: var(--radius-md) !important;
    font-family: var(--font) !important;
    animation: revealUp 400ms var(--ease-out) both;
}

/* ── EXPANDER ─────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--dur-base) ease;
}
div[data-testid="stExpander"]:hover { box-shadow: var(--shadow-md); }
div[data-testid="stExpander"] summary {
    font-family: var(--font) !important;
    font-weight: 600 !important;
}

/* ── SUBHEADERS ───────────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font) !important;
    font-weight: 700 !important;
    letter-spacing: -.025em !important;
}

/* ── PROGRESS BAR ─────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), #34a853) !important;
    border-radius: var(--radius-pill) !important;
    transition: width .6s var(--ease-out) !important;
}

/* ── KEYFRAME ANIMATIONS ──────────────────────────────────────────────────── */
@keyframes navDrop {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pageEnter {
    from { opacity: 0; transform: translateY(14px); filter: blur(6px); }
    to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
}
@keyframes revealUp {
    from { opacity: 0; transform: translateY(20px); filter: blur(4px); }
    to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
}
@keyframes heroEyebrow {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes heroTitle {
    from { opacity: 0; transform: translateY(28px) scale(.978); filter: blur(12px); }
    to   { opacity: 1; transform: translateY(0)    scale(1);    filter: blur(0); }
}
@keyframes heroSub {
    from { opacity: 0; transform: translateY(18px); filter: blur(6px); }
    to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
}
@keyframes resultReveal {
    from { opacity: 0; transform: translateY(24px) scale(.99); filter: blur(8px); }
    to   { opacity: 1; transform: translateY(0)    scale(1);   filter: blur(0); }
}
@keyframes ringIn {
    from { opacity: 0; transform: scale(.90); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .55; transform: scale(.85); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ── RESPONSIVE ───────────────────────────────────────────────────────────── */
@media (max-width: 900px) {
    .nav-links { display: none; }
    .hero h1   { font-size: clamp(2.8rem, 13vw, 5rem) !important; }
    .hero      { min-height: 70vh; padding: 8vh 1.2rem 7vh; }
    .hero-stats { gap: 1.5rem; }
    .result-grid { grid-template-columns: 1fr !important; text-align: center; }
    .contributors-grid { grid-template-columns: 1fr !important; }
    /* On mobile, full-bleed break-out doesn't work reliably — reset */
    .top-shell, .hero {
        width: 100% !important;
        margin-left: 0 !important;
    }
}

@media (max-width: 600px) {
    .hero h1   { font-size: clamp(2.3rem, 11vw, 4rem) !important; letter-spacing: -.03em !important; }
    .hero-stat-sep { display: none; }
    .hero-stats    { gap: 1rem; }
}

/* ── REDUCED MOTION ───────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .001ms !important;
    }
    .cursor { animation: none; opacity: 1; }
}

/* Always show revealed elements directly */
.pa-reveal {
    opacity: 1 !important;
    transform: none !important;
    filter: none !important;
}

</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  ·  consistent look across all charts
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Inter, ui-sans-serif, system-ui, sans-serif",
    font_color="#1c2030",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=48, b=16),
    title_font_size=15,
    title_font_color="#0d1117",
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        borderwidth=0,
        font_size=12,
    ),
    xaxis=dict(
        gridcolor="#f0f1f3",
        linecolor="#e5e7eb",
        tickfont_size=11,
        title_font_size=12,
    ),
    yaxis=dict(
        gridcolor="#f0f1f3",
        linecolor="#e5e7eb",
        tickfont_size=11,
        title_font_size=12,
    ),
)

PLOTLY_COLORS = ["#1a73e8", "#34a853", "#ea4335", "#fbbc05", "#a78bfa", "#06b6d4", "#f97316"]


def _fig(fig: go.Figure) -> go.Figure:
    """Apply the shared layout theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA / MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return train_and_persist()
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_training() -> dict:
    get_metrics.clear()
    return train_and_persist()


# ─────────────────────────────────────────────────────────────────────────────
# SCROLL-REVEAL  ·  inject once per page load via components.html
# ─────────────────────────────────────────────────────────────────────────────
def inject_scroll_reveal() -> None:
    """Inject IntersectionObserver-based scroll-reveal and count-up once."""
    components.html("""
    <script>
    (function(){
      if (window.__paRevealInit) return;
      window.__paRevealInit = true;

      /* ── Scroll-reveal ── */
      const style = document.createElement('style');
      style.textContent = `
        .pa-reveal {
          opacity: 1 !important;
          transform: none !important;
          filter: none !important;
          transition: opacity .6s cubic-bezier(.22,1,.36,1),
                      transform .6s cubic-bezier(.22,1,.36,1),
                      filter .6s cubic-bezier(.22,1,.36,1);
          will-change: opacity, transform, filter;
        }
        .pa-reveal.pa-visible {
          opacity: 1;
          transform: translateY(0);
          filter: blur(0);
        }
        @media (prefers-reduced-motion: reduce) {
          .pa-reveal { transition: none !important; opacity: 1 !important; transform: none !important; filter: none !important; }
        }
      `;
      document.head.appendChild(style);

      function revealAll() {
        const root = window.parent.document;
        const targets = root.querySelectorAll(
          '[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stMetric"]'
        );
        const io = new root.defaultView.IntersectionObserver(
          (entries) => {
            entries.forEach((e, i) => {
              if (e.isIntersecting) {
                setTimeout(() => {
                  e.target.classList.add('pa-reveal', 'pa-visible');
                }, i * 55);
                io.unobserve(e.target);
              }
            });
          },
          { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
        );
        targets.forEach(el => {
          if (!el.classList.contains('pa-visible')) {
            el.classList.add('pa-reveal');
            io.observe(el);
          }
        });
      }

      /* ── Count-up for metric values ── */
      function countUp(el, target, duration) {
        const isFloat = target % 1 !== 0;
        const decimals = isFloat ? (target.toString().split('.')[1] || '').length : 0;
        const suffix = el.textContent.replace(/[\d.,]/g, '').trim();
        const start = performance.now();
        function frame(now) {
          const p = Math.min((now - start) / duration, 1);
          const ease = 1 - Math.pow(1 - p, 4);
          const val = target * ease;
          el.textContent = val.toFixed(decimals) + suffix;
          if (p < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
      }

      function animateMetrics() {
        const root = window.parent.document;
        root.querySelectorAll('[data-testid="stMetricValue"]').forEach(el => {
          if (el.dataset.counted) return;
          el.dataset.counted = 'true';
          const raw = parseFloat(el.textContent.replace(/[^0-9.]/g, ''));
          if (!isNaN(raw) && raw > 0) countUp(el, raw, 900);
        });
      }

      // Run after Streamlit renders
      setTimeout(() => { revealAll(); animateMetrics(); }, 400);

      // Re-run on navigation (Streamlit re-renders)
      const observer = new MutationObserver(() => {
        setTimeout(() => { revealAll(); animateMetrics(); }, 300);
      });
      observer.observe(window.parent.document.body, { childList: true, subtree: true });
    })();
    </script>
    """, height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# LIGHT-MODE FLOATING PARTICLES  ·  matches screenshot — coloured dots on white
# ─────────────────────────────────────────────────────────────────────────────
def hero_particles_light() -> None:
    """Inject a lightweight canvas of scattered coloured dots over the white hero.
    Colours mirror the Google four-colour palette: blue, green, red, yellow.
    Particles drift slowly — no connections, no neural lines. GPU-friendly."""
    components.html("""
    <!doctype html>
    <html>
    <head>
    <style>
      html, body { margin:0; padding:0; background:transparent; overflow:hidden; }
      canvas {
        position: fixed;
        inset: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
      }
    </style>
    </head>
    <body>
    <canvas id="pc"></canvas>
    <script>
    (function(){
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

      const canvas = document.getElementById('pc');
      const ctx    = canvas.getContext('2d');

      /* Colours from the reference screenshot — blue, red, green, yellow/gold */
      const PALETTE = [
        'rgba(66,133,244,',   // Google Blue
        'rgba(234,67,53,',    // Google Red
        'rgba(52,168,83,',    // Google Green
        'rgba(251,188,5,',    // Google Yellow
        'rgba(100,149,237,',  // cornflower blue
        'rgba(180,100,220,',  // soft purple
      ];

      const N = 90;  // particle count — matches the density in the screenshot
      let W, H, pts = [];

      function rand(a, b) { return a + Math.random() * (b - a); }

      function make() {
        const col = PALETTE[Math.floor(Math.random() * PALETTE.length)];
        return {
          x:  rand(0, W),
          y:  rand(0, H),
          r:  rand(1.2, 3.8),          // small scattered dots
          vx: (Math.random() - .5) * .35,
          vy: (Math.random() - .5) * .28,
          a:  rand(.18, .55),          // alpha — semi-transparent like reference
          col,
        };
      }

      function init() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
        pts = Array.from({ length: N }, make);
      }

      function draw() {
        ctx.clearRect(0, 0, W, H);
        pts.forEach(p => {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          ctx.fillStyle = p.col + p.a + ')';
          ctx.fill();
        });
      }

      function update() {
        pts.forEach(p => {
          p.x += p.vx;
          p.y += p.vy;
          /* soft wrap — particle reappears on the opposite edge */
          if (p.x < -10) p.x = W + 10;
          if (p.x > W + 10) p.x = -10;
          if (p.y < -10) p.y = H + 10;
          if (p.y > H + 10) p.y = -10;
        });
      }

      let raf;
      function loop() { update(); draw(); raf = requestAnimationFrame(loop); }

      document.addEventListener('visibilitychange', () => {
        if (document.hidden) cancelAnimationFrame(raf);
        else loop();
      });

      window.addEventListener('resize', init);
      init();
      loop();
    })();
    </script>
    </body>
    </html>
    """, height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def top_bar(meta: dict, active_page: str) -> None:
    # ── Map the reference navbar items to project pages ────────────────────
    # Reference layout:  [Brand]  Products↓  Use Cases↓  Pricing  Enterprise  Resources↓   [Download↓]
    # Project mapping:
    #   Products     → Dashboard           (main product view)
    #   Use Cases    → Student Prediction  (primary use-case)
    #   Pricing      → Model Performance   (model metrics / "value")
    #   Enterprise   → Admin Analytics     (admin/institutional view)
    #   Resources    → About / Methodology (documentation)
    #   [pill/CTA]   → Dataset Explorer + Clustering (combined as Download-style pill)

    nav_items = [
        ("Products",          "Dashboard",            True),
        ("Use Cases",         "Student Prediction",   True),
        ("Data",              "Dataset Explorer",     False),
        ("Clustering",        "Clustering",           False),
        ("Performance",       "Model Performance",    False),
        ("Enterprise",        "Admin Analytics",      False),
        ("Resources",         "About / Methodology",  True),
    ]

    links_html = ""
    for display, page, has_chevron in nav_items:
        active_cls = "active" if active_page == page else ""
        chev = '<span class="chev">▾</span>' if has_chevron else ""
        links_html += (
            f'<a class="nav-link {active_cls}" '
            f'href="?nav={quote(page)}" target="_self">'
            f'{display}{chev}</a>'
        )

    # Four-colour Google-style A mark as inline SVG
    mark_svg = """
    <svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M11 2 L20 19 H2 Z" fill="none"/>
      <path d="M11 3.5 L4.5 17.5 H8 L11 11 L14 17.5 H17.5 Z" fill="#4285F4"/>
      <path d="M11 3.5 L17.5 17.5 H14 L11 11 Z" fill="#34A853"/>
      <path d="M7 13 H15" stroke="#FBBC05" stroke-width="2.2" stroke-linecap="round"/>
      <circle cx="11" cy="3.5" r="1.4" fill="#EA4335"/>
    </svg>
    """

    top_shell_html = f'''<div class="top-shell"><a class="brand" href="?nav=Dashboard" target="_self"><span class="brand-mark">{mark_svg}</span><span>Placement Antigravity</span></a><div class="nav-links">{links_html}</div><div class="nav-badge"><span>Best: {meta["model_name"]}</span><span class="arrow">↓</span></div></div>'''
    st.markdown(top_shell_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
def landing_hero(meta: dict, df: pd.DataFrame) -> None:
    placed_rate  = df[CONFIG.target].mean() * 100
    model_name   = meta["model_name"]
    best_f1      = meta["metrics"][model_name]["f1"]
    n_students   = len(df)

    # Four-colour mini mark for the eyebrow
    mini_mark = """
    <svg class="mini-mark" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 1.5 L13.5 13.5 H2.5 Z" fill="none"/>
      <path d="M8 2.5 L3.2 13 H5.5 L8 8 L10.5 13 H12.8 Z" fill="#4285F4"/>
      <path d="M8 2.5 L12.8 13 H10.5 L8 8 Z" fill="#34A853"/>
      <path d="M5 10 H11" stroke="#FBBC05" stroke-width="1.6" stroke-linecap="round"/>
      <circle cx="8" cy="2.5" r="1.1" fill="#EA4335"/>
    </svg>
    """

    st.markdown(
        _clean_html(f"""
        <section class="hero">
            <div class="hero-inner">
                <!-- Small brand badge above title — "△ Placement Antigravity" -->
                <div class="hero-eyebrow">
                    {mini_mark}
                    Placement Antigravity
                </div>

                <!-- Main title — large dark text like the reference -->
                <h1>
                    Student Placement<br>
                    <span class="gradient-word">Prediction</span> Platform<span class="cursor"></span>
                </h1>

                <!-- Single primary CTA — black pill like "Download" in the reference -->
                <div class="hero-actions">
                    <a class="hero-cta primary" href="?nav=Student%20Prediction" target="_self">
                        Predict Placement &nbsp;↓
                    </a>
                    <a class="hero-cta secondary" href="?nav=Admin%20Analytics" target="_self">
                        Explore Analytics
                    </a>
                </div>

                <!-- Stat strip -->
                <div class="hero-stats">
                    <div class="hero-stat">
                        <div class="hero-stat-value">{n_students:,}</div>
                        <div class="hero-stat-label">Students</div>
                    </div>
                    <div class="hero-stat-sep"></div>
                    <div class="hero-stat">
                        <div class="hero-stat-value">{placed_rate:.0f}%</div>
                        <div class="hero-stat-label">Placement Rate</div>
                    </div>
                    <div class="hero-stat-sep"></div>
                    <div class="hero-stat">
                        <div class="hero-stat-value">7</div>
                        <div class="hero-stat-label">ML Models</div>
                    </div>
                    <div class="hero-stat-sep"></div>
                    <div class="hero-stat">
                        <div class="hero-stat-value">{best_f1:.2f}</div>
                        <div class="hero-stat-label">Best F1 Score</div>
                    </div>
                </div>
            </div>
        </section>
        """),
        unsafe_allow_html=True,
    )

    # Inject light-mode floating particles on top of the white hero
    hero_particles_light()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE SECTION HEADER
# ─────────────────────────────────────────────────────────────────────────────
def header(title: str, subtitle: str, pill: str = "ML Platform") -> None:
    st.markdown(
        _clean_html(f"""
        <div class="page-hero">
            <span class="pill">{pill}</span>
            <h1>{title}</h1>
            <p class="subtle">{subtitle}</p>
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.write("")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for model, values in metrics.items():
        rows.append({
            "Model":     model,
            "Accuracy":  values.get("accuracy"),
            "Precision": values.get("precision"),
            "Recall":    values.get("recall"),
            "F1":        values.get("f1"),
            "ROC-AUC":   values.get("roc_auc"),
            "CV F1":     values.get("cv_f1"),
        })
    return pd.DataFrame(rows).sort_values(["F1", "ROC-AUC"], ascending=False)


def _section_kicker(text: str) -> None:
    st.markdown(f"<span class='section-kicker'>{text}</span>", unsafe_allow_html=True)


def _divider() -> None:
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1.8rem 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOADING ANIMATION  ·  injected during prediction wait
# ─────────────────────────────────────────────────────────────────────────────
def prediction_loading_html() -> str:
    return """
    <div class="loading-overlay">
        <div class="loading-spinner"></div>
        <div class="loading-steps">
            <div class="loading-step done"   id="s1"><span class="step-dot"></span>Validating student profile&hellip;</div>
            <div class="loading-step active" id="s2"><span class="step-dot"></span>Engineering features&hellip;</div>
            <div class="loading-step"        id="s3"><span class="step-dot"></span>Running ML model&hellip;</div>
            <div class="loading-step"        id="s4"><span class="step-dot"></span>Generating prediction&hellip;</div>
            <div class="loading-step"        id="s5"><span class="step-dot"></span>Building recommendations&hellip;</div>
        </div>
    </div>
    <script>
    (function() {
        const steps = ['s1','s2','s3','s4','s5'];
        let i = 2;
        function advance() {
            if (i >= steps.length) return;
            const prev = document.getElementById(steps[i - 1]);
            if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
            const curr = document.getElementById(steps[i]);
            if (curr) curr.classList.add('active');
            i++;
            setTimeout(advance, 600 + Math.random() * 400);
        }
        setTimeout(advance, 500);
    })();
    </script>
    """


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULT CARD
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction_result(result: dict, row: dict) -> None:
    prob       = result["placement_probability"]
    label      = result["label"]
    confidence = result["confidence"]
    placed     = result["prediction"] == 1
    readiness  = result["readiness"]
    recs       = result["recommendations"]
    contribs   = local_contributors(row)

    title_cls  = "placed" if placed else "risk"
    ring_color = "#1a73e8" if placed else "#ea4335"
    conf_tag   = "tag-green" if confidence == "High" else "tag-amber" if confidence == "Medium" else "tag-red"
    ready_icon = "🟢" if readiness["score"] >= 75 else "🟡" if readiness["score"] >= 55 else "🔴"

    # Ring CSS — uses conic-gradient with the probability
    p_css = min(max(prob, 0), 100)
    ring_gradient = (
        f"radial-gradient(circle closest-side, white 69%, transparent 70%),"
        f"conic-gradient({ring_color} {p_css}%, #e8eef7 0)"
    )

    # Contributor bars
    def build_contrib_rows(items: list, bar_cls: str, max_val: float) -> str:
        if not items:
            return "<p style='color:#9ca3af;font-size:.85rem;'>None detected.</p>"
        html = ""
        for feat, score in items:
            pct = min(score / max(max_val, 1) * 100, 100)
            label_nice = feat.replace("_", " ").title()
            html += f"""
            <div class="contributor-row">
                <span style="min-width:140px;font-size:.82rem">{label_nice}</span>
                <div class="contributor-bar-wrap">
                    <div class="contributor-bar {bar_cls}" style="width:{pct:.1f}%"></div>
                </div>
                <span style="font-size:.8rem;font-weight:700;color:#374151">+{score:.1f}%</span>
            </div>"""
        return html

    max_pos  = max((s for _, s in contribs["positive"]),  default=1)
    max_impr = max((s for _, s in contribs["improvement"]),default=1)
    pos_html  = build_contrib_rows(contribs["positive"],   "positive", max_pos)
    impr_html = build_contrib_rows(contribs["improvement"],"improve",  max_impr)

    # Recommendation items
    rec_icons = ["📚","💻","🏆","🎯","🌐","🤝","🔧","📊"]
    recs_html = ""
    for i, rec in enumerate(recs):
        icon = rec_icons[i % len(rec_icons)]
        recs_html += f"""
        <div class="rec-item" style="animation-delay:{i * .08}s">
            <div class="rec-item-icon">{icon}</div>
            <span>{rec}</span>
        </div>"""

    # Readiness breakdown bars
    breakdown = readiness["breakdown"]
    bd_html = ""
    for group, score in breakdown.items():
        color = "#1a73e8" if score >= 75 else "#f59e0b" if score >= 55 else "#ef4444"
        bd_html += f"""
        <div style="margin-bottom:.55rem">
            <div style="display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:.2rem">
                <span style="color:#374151;font-weight:500">{group}</span>
                <span style="font-weight:700;color:{color}">{score:.0f}</span>
            </div>
            <div style="height:6px;background:#e5e7eb;border-radius:99px;overflow:hidden">
                <div style="height:100%;width:{score}%;background:{color};border-radius:99px;
                            transition:width .9s cubic-bezier(.22,1,.36,1)"></div>
            </div>
        </div>"""

    st.markdown(
        _clean_html(f"""
        <div class="result-card">
            <div class="result-grid">
                <!-- Ring -->
                <div>
                    <div class="prob-ring" style="background:{ring_gradient}">
                        <div class="prob-ring-inner">
                            <strong>{prob:.1f}%</strong>
                            <span>Probability</span>
                        </div>
                    </div>
                </div>
                <!-- Details -->
                <div>
                    <h2 class="result-title {title_cls}">{label}</h2>
                    <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-top:.4rem">
                        <span class="tag {conf_tag}">{confidence} Confidence</span>
                        <span style="color:#6b7280;font-size:.82rem">· ML probability based on {result.get("prediction","")!s}</span>
                    </div>
                    <div class="readiness-pill">
                        {ready_icon} Readiness: <strong>{readiness['score']:.0f} / 100</strong>
                        &nbsp;·&nbsp; {readiness['category']}
                    </div>
                    <!-- Readiness breakdown -->
                    <div style="margin-top:1.2rem">{bd_html}</div>
                </div>
            </div>

            <!-- Contributors -->
            <div class="contributors-grid" style="margin-top:1.8rem">
                <div class="contributor-panel">
                    <h4>✅ Positive Factors</h4>
                    {pos_html}
                </div>
                <div class="contributor-panel">
                    <h4>⚡ Improvement Areas</h4>
                    {impr_html}
                </div>
            </div>

            <!-- Recommendations -->
            <div style="margin-top:1.8rem">
                <h4 style="font-size:.82rem;font-weight:800;letter-spacing:.06em;
                           text-transform:uppercase;color:#6b7280;margin:0 0 .7rem">
                    🧭 Recommended Actions
                </h4>
                <div class="recs-list">{recs_html}</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────
def dashboard(df: pd.DataFrame, meta: dict) -> None:
    landing_hero(meta, df)

    # ── Live Metrics ──
    st.markdown("<br>", unsafe_allow_html=True)
    _section_kicker("Live Placement Intelligence")
    placed_rate   = df[CONFIG.target].mean() * 100
    avg_skill     = df[["coding_score","dsa_score","aptitude_score","communication_score"]].mean(axis=1).mean()
    readiness_scores = df.apply(lambda r: calculate_readiness(r)["score"], axis=1)
    high_risk     = int((readiness_scores < 55).sum())

    cols = st.columns(6)
    cols[0].metric("Students",        f"{len(df):,}")
    cols[1].metric("Placement Rate",  f"{placed_rate:.1f}%")
    cols[2].metric("Avg CGPA",        f"{df['cgpa'].mean():.2f}")
    cols[3].metric("Avg Skill",       f"{avg_skill:.1f}")
    cols[4].metric("High Risk",       f"{high_risk}")
    cols[5].metric("Best Model",      meta["model_name"])

    _divider()

    # ── Charts row 1 ──
    _section_kicker("Placement Distribution & Academic Signals")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x="placed", color="placed",
            title="Placement Distribution",
            text_auto=True,
            color_discrete_map={0: "#ef4444", 1: "#34a853"},
            labels={"placed": "Placement Status"},
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(_fig(fig), use_container_width=True)

    with c2:
        fig = px.scatter(
            df, x="cgpa", y="coding_score",
            color=df["placed"].map({1: "Placed", 0: "Not Placed"}),
            title="CGPA vs Coding Score",
            color_discrete_map={"Placed": "#1a73e8", "Not Placed": "#ef4444"},
            opacity=.72,
        )
        st.plotly_chart(_fig(fig), use_container_width=True)

    # ── Charts row 2 ──
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(
            df, x=df["placed"].map({1: "Placed", 0: "Not Placed"}),
            y="cgpa", color=df["placed"].map({1: "Placed", 0: "Not Placed"}),
            title="CGPA Distribution by Outcome",
            color_discrete_map={"Placed": "#1a73e8", "Not Placed": "#ef4444"},
        )
        st.plotly_chart(_fig(fig), use_container_width=True)

    with c4:
        mtable = metrics_table(meta["metrics"])
        fig = px.bar(
            mtable, x="Model", y="F1",
            title="Model F1 Comparison",
            color="F1",
            color_continuous_scale=[[0,"#dbeafe"],[.5,"#1a73e8"],[1,"#1e40af"]],
            text_auto=".3f",
        )
        fig.update_traces(textfont_size=11, marker_line_width=0)
        fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-18)
        st.plotly_chart(_fig(fig), use_container_width=True)

    _divider()

    # ── Feature Importance ──
    _section_kicker("Global Feature Importance")
    try:
        importance = global_feature_importance(df)
        fig = px.bar(
            importance, x="importance", y="feature", orientation="h",
            title="Permutation Importance (F1 Scoring)",
            color="importance",
            color_continuous_scale=[[0,"#dbeafe"],[.5,"#1a73e8"],[1,"#1e40af"]],
        )
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", tickfont_size=12),
        )
        st.plotly_chart(_fig(fig), use_container_width=True)
    except Exception as exc:
        st.info(f"Feature importance will be available after retraining: {exc}")


def student_prediction(df: pd.DataFrame) -> None:
    header(
        "Student Prediction",
        "Enter a student profile to estimate placement likelihood and receive personalised improvement actions.",
        pill="Prediction Engine",
    )

    branches = sorted(df["branch"].dropna().unique())
    specs    = sorted(df["specialization"].dropna().unique())

    with st.form("prediction_form"):
        _section_kicker("Academic Profile")
        a, b, c = st.columns(3)
        cgpa         = a.slider("CGPA",         4.0, 10.0, 7.2, 0.1, help="Cumulative Grade Point Average")
        attendance   = b.slider("Attendance %", 40.0, 100.0, 78.0, 1.0)
        backlogs     = c.number_input("Backlogs", min_value=0, max_value=10, value=0)
        branch       = a.selectbox("Branch",         branches)
        specialization = b.selectbox("Specialization", specs)

        st.write("")
        _section_kicker("Skills & Experience")
        s1, s2, s3, s4 = st.columns(4)
        coding       = s1.slider("Coding Score",       0.0, 100.0, 70.0, 1.0)
        dsa          = s2.slider("DSA Score",          0.0, 100.0, 65.0, 1.0)
        aptitude     = s3.slider("Aptitude Score",     0.0, 100.0, 68.0, 1.0)
        communication= s4.slider("Communication",      0.0, 100.0, 66.0, 1.0)
        interview    = s1.slider("Interview Score",    0.0, 100.0, 66.0, 1.0)
        projects     = s2.number_input("Projects",      min_value=0, max_value=10,  value=2)
        certifications = s3.number_input("Certifications", min_value=0, max_value=10, value=2)
        internships  = s4.number_input("Internships",   min_value=0, max_value=5,   value=1)
        hackathons   = s1.number_input("Hackathons",    min_value=0, max_value=10,  value=1)

        submitted = st.form_submit_button("🎯 Predict Placement", use_container_width=True)

    row = {
        "branch": branch, "specialization": specialization,
        "cgpa": cgpa, "attendance_pct": attendance, "backlogs": backlogs,
        "coding_score": coding, "dsa_score": dsa, "aptitude_score": aptitude,
        "communication_score": communication, "interview_score": interview,
        "projects": projects, "certifications": certifications,
        "internships": internships, "hackathons": hackathons,
    }

    if submitted:
        # Show animated loading state while computing
        loading_placeholder = st.empty()
        loading_placeholder.markdown(prediction_loading_html(), unsafe_allow_html=True)

        result = predict_student(row)   # actual ML call

        loading_placeholder.empty()

        st.markdown("<br>", unsafe_allow_html=True)
        render_prediction_result(result, row)

        # Readiness chart — polished
        st.markdown("<br>", unsafe_allow_html=True)
        _section_kicker("Readiness Breakdown")
        breakdown = result["readiness"]["breakdown"]
        fig = px.bar(
            x=list(breakdown.values()),
            y=list(breakdown.keys()),
            orientation="h",
            title="Placement Readiness by Area",
            color=list(breakdown.values()),
            color_continuous_scale=[[0,"#fee2e2"],[.5,"#fef3c7"],[1,"#dcfce7"]],
            range_x=[0, 100],
            text=[f"{v:.0f}" for v in breakdown.values()],
        )
        fig.update_traces(textfont_size=12, textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", tickfont_size=13),
        )
        st.plotly_chart(_fig(fig), use_container_width=True)


def dataset_explorer(df: pd.DataFrame) -> None:
    header(
        "Dataset Explorer",
        "Inspect data quality, distributions, missing values, duplicates, correlations, and outliers.",
        pill="Data Insights",
    )
    summary = dataset_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           summary["shape"][0])
    c2.metric("Columns",        summary["shape"][1])
    c3.metric("Duplicate Rows", summary["duplicate_rows"])
    missing_count = sum(v for v in summary["missing_values"].values() if v > 0)
    c4.metric("Missing Values", missing_count)

    _divider()

    _section_kicker("Data Preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        _section_kicker("Missing Values")
        mv = {k: v for k, v in summary["missing_values"].items() if v > 0}
        if mv:
            st.dataframe(pd.DataFrame(mv.items(), columns=["Column","Missing"]), use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values detected.")

    with c2:
        _section_kicker("IQR Outlier Counts")
        st.dataframe(
            pd.DataFrame(summary["outlier_counts"].items(), columns=["Feature","IQR Outliers"]),
            use_container_width=True, hide_index=True,
        )

    _divider()
    _section_kicker("Statistical Summary")
    st.dataframe(summary["statistics"], use_container_width=True)

    _section_kicker("Correlation Matrix")
    corr = df.select_dtypes("number").corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
    )
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    st.plotly_chart(_fig(fig), use_container_width=True)


def model_performance(meta: dict) -> None:
    header(
        "Model Performance",
        "Training status, comparative metrics, evaluation charts, and regression results.",
        pill="Model Lab",
    )

    if st.button("🔄 Train Models"):
        with st.spinner("Training all ML models with cross-validation…"):
            meta = refresh_training()
        st.success(f"✅ Training complete. Best model: **{meta['model_name']}**")

    c1, c2 = st.columns(2)
    c1.markdown(f"**Model version:** `{meta['created_at']}`")
    c2.markdown(f"**Selection metric:** `{meta['selection_metric']}`")

    _divider()
    _section_kicker("All Models — Comparison")
    table = metrics_table(meta["metrics"])
    st.dataframe(table, use_container_width=True, hide_index=True)

    best   = meta["model_name"]
    metric = meta["metrics"][best]

    st.info(f"🏆 **Selected model: {best}** — chosen by F1 score first, then ROC-AUC as tie-breaker.")

    c1, c2 = st.columns(2)

    with c1:
        _section_kicker("Confusion Matrix")
        cm  = metric["confusion_matrix"]
        fig = px.imshow(
            cm, text_auto=True,
            labels={"x":"Predicted","y":"Actual","color":"Count"},
            title=f"{best} — Confusion Matrix",
            color_continuous_scale=[[0,"#f9fafb"],[1,"#1a73e8"]],
        )
        st.plotly_chart(_fig(fig), use_container_width=True)

    with c2:
        _section_kicker("ROC Curve")
        roc = metric["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines", name=best,
            line=dict(color="#1a73e8", width=2.5),
            fill="tozeroy", fillcolor="rgba(26,115,232,.06)",
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Random",
            line=dict(color="#9ca3af", dash="dash", width=1.5),
        ))
        fig.update_layout(
            title=f"ROC Curve — AUC {metric['roc_auc']:.4f}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(_fig(fig), use_container_width=True)

    if meta.get("regression"):
        _divider()
        _section_kicker("Optional Salary Regression (placed students only)")
        st.info("Regression is trained only on placed students — salary / package is a continuous target.")
        reg_df = pd.DataFrame(meta["regression"]["metrics"]).T.reset_index()
        reg_df.columns = ["Model"] + list(reg_df.columns[1:])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)


def clustering_page(df: pd.DataFrame) -> None:
    header(
        "Student Segmentation",
        "K-Means clustering groups students by profile similarity; descriptions are derived from cluster statistics.",
        pill="Clustering",
    )

    with st.spinner("Running K-Means analysis…"):
        result = run_clustering(df)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            pd.DataFrame(result["elbow"]), x="k", y="inertia",
            markers=True, title="Elbow Method — Inertia vs K",
        )
        fig.update_traces(line_color="#1a73e8", marker_color="#1a73e8")
        st.plotly_chart(_fig(fig), use_container_width=True)

    with c2:
        fig = px.line(
            pd.DataFrame(result["silhouette"]), x="k", y="silhouette",
            markers=True, title="Silhouette Score vs K",
        )
        fig.update_traces(line_color="#34a853", marker_color="#34a853")
        st.plotly_chart(_fig(fig), use_container_width=True)

    _divider()
    _section_kicker("Cluster Profiles")
    st.dataframe(result["profiles"], use_container_width=True, hide_index=True)

    for item in result["descriptions"]:
        color = PLOTLY_COLORS[item["cluster"] % len(PLOTLY_COLORS)]
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:.6rem;
                            margin:.3rem 0;padding:.55rem .9rem;
                            border-radius:8px;border:1px solid #e5e7eb;
                            background:#f9fafb;font-size:.9rem">
                    <span style="width:10px;height:10px;border-radius:50%;
                                 background:{color};flex-shrink:0"></span>
                    <strong>Cluster {item['cluster']}</strong>
                    &nbsp;·&nbsp; {item['description']}
                    <span style="margin-left:auto;color:#6b7280;font-size:.8rem">
                        {item['student_count']} students</span>
                </div>""",
            unsafe_allow_html=True,
        )

    _divider()
    _section_kicker("Cluster Visualization")
    clustered = result["data"]
    fig = px.scatter(
        clustered, x="cgpa", y="coding_score",
        color=clustered["cluster"].astype(str),
        size="projects", size_max=14,
        hover_data=["student_id","branch"],
        title="Clusters — CGPA vs Coding Score",
        color_discrete_sequence=PLOTLY_COLORS,
        opacity=.78,
    )
    st.plotly_chart(_fig(fig), use_container_width=True)


def analytics(df: pd.DataFrame) -> None:
    header(
        "Admin Analytics",
        "Filter students by branch and CGPA range, inspect risk bands, and identify intervention priorities.",
        pill="Admin",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        branches   = st.multiselect(
            "Branch Filter", sorted(df["branch"].unique()),
            default=sorted(df["branch"].unique()),
        )
    with c2:
        cgpa_range = st.slider("CGPA Range", 4.0, 10.0, (4.0, 10.0), 0.1)

    filtered = df[df["branch"].isin(branches) & df["cgpa"].between(*cgpa_range)].copy()
    filtered["readiness_score"]    = filtered.apply(lambda r: calculate_readiness(r)["score"],    axis=1)
    filtered["readiness_category"] = filtered.apply(lambda r: calculate_readiness(r)["category"], axis=1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Students",  len(filtered))
    m2.metric("Placement Rate",     f"{filtered['placed'].mean()*100:.1f}%" if len(filtered) else "0%")
    m3.metric("Avg Readiness",      f"{filtered['readiness_score'].mean():.1f}" if len(filtered) else "0")
    m4.metric("High-Risk Students", int((filtered["readiness_score"] < 55).sum()) if len(filtered) else 0)

    _divider()

    c1, c2 = st.columns(2)
    with c1:
        _section_kicker("Branch-wise Placement Rate")
        branch_rate = (
            filtered.groupby("branch")["placed"]
            .mean().mul(100).reset_index()
            .rename(columns={"placed": "Placement Rate %"})
            .sort_values("Placement Rate %", ascending=False)
        )
        fig = px.bar(
            branch_rate, x="branch", y="Placement Rate %",
            title="Placement Rate by Branch",
            color="Placement Rate %",
            color_continuous_scale=[[0,"#fee2e2"],[.5,"#fef3c7"],[1,"#dcfce7"]],
            text_auto=".1f",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_fig(fig), use_container_width=True)

    with c2:
        _section_kicker("Readiness Distribution")
        fig = px.histogram(
            filtered, x="readiness_score",
            nbins=20,
            title="Readiness Score Distribution",
            color_discrete_sequence=["#1a73e8"],
        )
        fig.add_vline(x=55, line_dash="dash", line_color="#ef4444",
                      annotation_text="Risk Threshold", annotation_position="top right")
        st.plotly_chart(_fig(fig), use_container_width=True)

    _divider()
    _section_kicker("Intervention Priority List — Bottom 30 by Readiness")
    risk = filtered.sort_values("readiness_score").head(30).copy()
    risk["weak_areas"]              = risk.apply(lambda r: ", ".join(weak_features(r, 3)), axis=1)
    risk["recommended_intervention"]= risk.apply(lambda r: generate_recommendations(r, 1)[0], axis=1)
    st.dataframe(
        risk[["student_id","branch","cgpa","placed","readiness_score",
              "readiness_category","weak_areas","recommended_intervention"]],
        use_container_width=True,
        hide_index=True,
    )


def methodology() -> None:
    header(
        "About & Methodology",
        "A viva-friendly explanation of every ML choice, responsible AI decisions, and project context.",
        pill="Methodology",
    )

    sections = [
        ("📊 Dataset",
         "A clearly labelled **Demonstration/Synthetic Dataset** (750 rows × 19 columns) "
         "generated for academic ML demonstration — no real institutional data was used. "
         "The generation formula uses a logistic function of all feature scores with a branch-based boost."),
        ("🔧 Preprocessing",
         "Missing numeric values are **median-imputed** and **StandardScaler-normalised**; "
         "categorical values are **most-frequent-imputed** and **one-hot encoded** inside an sklearn "
         "`Pipeline` to prevent data leakage between training folds and the test set."),
        ("🤖 Classification",
         "The primary target is `placed` (binary). Six models are trained: "
         "Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, "
         "and a Calibrated SVM. A soft **Voting Ensemble** (LR + RF + GB) is also evaluated. "
         "GridSearchCV with StratifiedKFold(5) tunes each model."),
        ("📈 Probabilistic Learning",
         "Placement probability is produced by classifiers with `predict_proba`. SVM uses "
         "`CalibratedClassifierCV` with sigmoid calibration to enable probability output. "
         "Confidence bands: **High** (≥75% or ≤25%), **Medium** (|p−0.5| ≥ 0.12), **Low** otherwise."),
        ("💰 Regression",
         "Package prediction is an optional regression task trained **only on placed students** "
         "because salary is a continuous variable only defined for placed outcomes. "
         "Linear Regression and Ridge Regression are compared by R²."),
        ("🔵 Clustering",
         "K-Means segments students into profile groups. Silhouette score and elbow method "
         "guide k-selection (default k=4). Cluster descriptions are auto-generated from "
         "the top-3 highest-mean features per cluster — not hardcoded."),
        ("📏 Evaluation",
         "Accuracy, Precision, Recall, **F1**, ROC-AUC, confusion matrix, and ROC curve are reported. "
         "**F1 is used for model selection** because both false positives (unfair labels) and "
         "false negatives (missed interventions) have real costs."),
        ("🔍 Explainability",
         "Global: **permutation importance** (5 repeats, F1 scoring). "
         "Local: perturbation-based — each feature is reduced to 75% of its value and the "
         "probability change is measured. This is an approximation, not SHAP."),
        ("⚖️ Responsible AI",
         "**Gender** is present in the dataset for responsible-AI discussion but is in "
         "`excluded_from_model` — it never enters the feature matrix. "
         "All classifiers with `class_weight` support use `class_weight='balanced'` "
         "to account for the 73/27% class imbalance. "
         "Outputs are clearly labelled as ML estimates, not guarantees."),
        ("🌍 SDG Alignment",
         "SDG 4 Quality Education · SDG 8 Decent Work and Economic Growth · "
         "SDG 9 Industry, Innovation and Infrastructure."),
    ]

    for icon_title, body in sections:
        with st.expander(icon_title, expanded=False):
            st.markdown(body)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Disclaimer:** This system provides machine-learning-based estimates "
        "for educational and analytical purposes only. It does not guarantee "
        "employment or placement outcomes."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df         = get_data()
    validation = validate_dataset(df)
    if not validation["is_valid"]:
        st.error(f"Dataset validation failed: {validation}")
        st.stop()

    meta = get_metrics()

    pages = [
        "Dashboard",
        "Student Prediction",
        "Admin Analytics",
        "Dataset Explorer",
        "Model Performance",
        "Clustering",
        "About / Methodology",
    ]

    # Handle URL-based navigation
    requested_page = st.query_params.get("nav")
    if requested_page in pages:
        st.session_state["selected_page"] = requested_page
        st.query_params.clear()

    if "selected_page" not in st.session_state or st.session_state["selected_page"] not in pages:
        st.session_state["selected_page"] = "Dashboard"

    # Navbar (HTML — shows active page)
    top_bar(meta, active_page=st.session_state["selected_page"])

    # Streamlit radio for actual page switching
    page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state["selected_page"]),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["selected_page"] = page

    # Inject scroll-reveal + count-up once per page
    inject_scroll_reveal()

    # Route
    if page == "Dashboard":
        dashboard(df, meta)
    elif page == "Student Prediction":
        student_prediction(df)
    elif page == "Admin Analytics":
        analytics(df)
    elif page == "Dataset Explorer":
        dataset_explorer(df)
    elif page == "Model Performance":
        model_performance(meta)
    elif page == "Clustering":
        clustering_page(df)
    else:
        methodology()


if __name__ == "__main__":
    main()
