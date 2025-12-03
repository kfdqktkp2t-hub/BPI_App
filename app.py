import io, re, math, textwrap, json, zipfile, datetime as dt
from dateutil import tz
from dateutil.parser import parse as parse_dt
from typing import Dict, List, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

from PIL import Image, ImageDraw, ImageFont

import streamlit as st

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="CBB Matchups: ESPN BPI", page_icon="🏀", layout="wide")

PRIMARY = "#0A2540"    # deep navy
ACCENT  = "#14B8A6"    # teal-ish
ACCENT2 = "#F97316"    # orange
LIGHT   = "#F8FAFC"    # near-white
DARK    = "#0B1020"

TIMEZONE = "America/New_York"   # your local for display
DIV1_GROUP_ID = "50"            # ESPN uses 50 for D1 on many NCAAM pages

# Common alias fixes between sources
TEAM_ALIASES: Dict[str, str] = {
    "UConn": "Connecticut",
    "Ole Miss": "Mississippi",
    "Miami (FL)": "Miami (FL)",
    "Miami": "Miami (FL)",  # ESPN often uses (FL) for NCAAM
    "UTSA": "UT San Antonio",
    "UC Santa Barbara": "Santa Barbara",
    "Central Florida": "UCF",
    "Southern California": "USC",
    "UNC": "North Carolina",
    "Alabama-Birmingham": "UAB",
}

# -----------------------
# Utilities
# -----------------------
@st.cache_data(ttl=3*60*60, show_spinner=False)
def fetch_espn_scoreboard(date_yyyymmdd: str) -> dict:
    """
    ESPN men's CBB scoreboard JSON (undocumented public endpoint).
    Example base: https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {"dates": date_yyyymmdd, "groups": DIV1_GROUP_ID, "limit": 500}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _clean_team_name(name: str) -> str:
    name = name.replace(" St.", " State").replace(" Univ.", " University").strip()
    return TEAM_ALIASES.get(name, name)

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_table(season_year: int) -> pd.DataFrame:
    """
    Pulls the BPI table directly from ESPN HTML and parses it into a DataFrame.
    If ESPN changes HTML, code falls back to a more defensive extraction.
    """
    url = "https://www.espn.com/mens-college-basketball/bpi"
    # NOTE: ESPN displays current season by default. We’ll filter by season if present on page.
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=20).text
    tables = pd.read_html(html, flavor="lxml")  # parse all tables found

    # Find the main BPI table (has columns like 'Team', 'BPI', 'BPI RK', 'OFF', 'DEF')
    cand = []
    for t in tables:
        cols = [c.lower() for c in t.columns.map(lambda x: str(x))] if hasattr(t, "columns") else []
        if any("bpi" in c for c in cols) and any("team" in c for c in cols):
            cand.append(t.copy())

    if not cand:
        # Defensive parse: try scraping via <table> search in the HTML
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if table is None:
            raise RuntimeError("Could not find BPI table on ESPN page.")
        df = pd.read_html(str(table))[0]
    else:
        # pick the widest reasonable table
        df = max(cand, key=lambda d: d.shape[1])

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # Some ESPN tables have multi-level headers; normalize common names
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c] = "Team"
        elif cl == "bpi": ren[c] = "BPI"
        elif "bpi rk" in cl or (("rk" in cl) and "bpi" in cl): ren[c] = "BPI_RK"
        elif cl.startswith("off"): ren[c] = "OFF"
        elif cl.startswith("def"): ren[c] = "DEF"
        elif "w-l" in cl and "conf" not in cl: ren[c] = "W_L"
    if ren:
        df = df.rename(columns=ren)

    # Only keep what we need
    keep_cols = [c for c in ["Team", "BPI", "BPI_RK", "OFF", "DEF"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["Team"]).copy()

    # Clean values
    def to_num(x):
        try:
            return float(str(x).strip().replace(",", ""))
        except:
            return None
    if "BPI" in df.columns:
        df["BPI"] = df["BPI"].apply(to_num)

    # Team name cleanup
    df["Team"] = df["Team"].astype(str).str.replace(r"\s+\(\d+\)$", "", regex=True).str.strip()
    df["Team_norm"] = df["Team"].apply(_clean_team_name)

    # Deduplicate if the page shows multiple chunks
    df = df.drop_duplicates(subset=["Team_norm"]).reset_index(drop=True)

    return df

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_torvik_rating(year: int) -> pd.DataFrame:
    """
    Optional fallback (or comparison): pulls Torvik advanced team stats CSV.
    Officially encouraged by Bart Torvik to use programmatically.
    """
    url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
    df = pd.read_csv(url, header=None)
    # First row in this CSV is headers in many seasons; if not, adjust as needed.
    # Try to detect header row (it often starts with 'Team' in col 0)
    if isinstance(df.iloc[0,0], str) and df.iloc[0,0].lower().startswith("team"):
        df.columns = df.iloc[0]
        df = df[1:]
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Torvik rating column is "ADJEM" (Adj Efficiency Margin). We'll map it as TRank.
    if "Team" not in df.columns:
        # older seasons: team name often at col 0
        df = df.rename(columns={df.columns[0]: "Team"})
    df["Team_norm"] = df["Team"].apply(_clean_team_name)
    if "ADJEM" in df.columns:
        df["TRank"] = pd.to_numeric(df["ADJEM"], errors="coerce")
    else:
        df["TRank"] = pd.NA
    return df[["Team", "Team_norm", "TRank"]].dropna(subset=["Team_norm"]).reset_index(drop=True)

def best_match(name: str, candidates: List[str]) -> str:
    # fuzzy match with RapidFuzz; fall back to difflib via process.extractOne
    match = process.extractOne(name, candidates, scorer=fuzz.WRatio)
    return match[0] if match else name

def team_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except:
        return ImageFont.load_default()

def draw_card(game_row: pd.Series) -> Image.Image:
    """
    Create a 1600x900 image card for X posts.
    """
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), color=(11,16,32))  # DARK
    d = ImageDraw.Draw(img)

    # diagonal accent
    d.polygon([(0, H*0.65), (W*0.6, H), (0, H)], fill=(20,184,166))  # ACCENT
    d.polygon([(W, H*0.35), (W, H), (W*0.4, H)], fill=(249,115,22))  # ACCENT2

    # Text blocks
    title_font = team_font(56)
    sub_font   = team_font(40)
    big_font   = team_font(110)
    med_font   = team_font(60)

    # Header
    header = f"{game_row['date_str']}  •  ESPN BPI"
    d.text((60, 40), header, fill=(248,250,252), font=sub_font)

    # Teams
    left = 80
    top = 160
    d.text((left, top), game_row["away"], fill=(255,255,255), font=title_font)
    d.text((left, top+80), f"BPI {game_row['BPI_away']:+.1f}", fill=(255,255,255), font=med_font)

    d.text((left, top+210), "@", fill=(200,205,220), font=med_font)

    d.text((left, top+300), game_row["home"], fill=(255,255,255), font=title_font)
    d.text((left, top+380), f"BPI {game_row['BPI_home']:+.1f}", fill=(255,255,255), font=med_font)

    # Gap
    gap = game_row["BPI_diff"]
    gap_txt = f"Rating Gap: {gap:+.1f}"
    d.text((W-700, 160), gap_txt, fill=(255,255,255), font=big_font)

    # Footer
    footer = f"{game_row['time_str']} ET • {game_row.get('location','').strip()}"
    d.text((60, H-90), footer, fill=(12,19,36), font=title_font)

    return img

# -----------------------
# Core app
# -----------------------
st.markdown(
    f"""
    <div style="padding:1rem;border-radius:12px;background:{PRIMARY};color:{LIGHT}">
      <h2 style="margin:0">🏀 CBB Matchups • ESPN BPI</h2>
      <div style="opacity:.85">Daily games, BPI ratings, gap highlight, and exportable social cards.</div>
    </div>
    """, unsafe_allow_html=True
)

colA, colB, colC = st.columns([1,1,1])
with colA:
    dflt = dt.datetime.now(tz.gettz(TIMEZONE)).date()
    date = st.date_input("Date", dflt)
with colB:
    use_fallback = st.checkbox("Also load Torvik rating (fallback/compare)", False)
with colC:
    card_limit = st.number_input("Max cards to export", min_value=1, max_value=200, value=24, step=1)

date_str = date.strftime("%Y%m%d")

with st.spinner("Fetching games…"):
    score = fetch_espn_scoreboard(date_str)

events = score.get("events", [])
if not events:
    st.info("No Division I games found for this date.")
    st.stop()

# Build games list
games = []
for e in events:
    comp = e.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        continue
    # ESPN flips home/away sometimes — normalize by 'homeAway'
    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

    def get_team_name(c):
        nm = c.get("team", {}).get("location") or c.get("team", {}).get("shortDisplayName") or c.get("team", {}).get("displayName")
        return _clean_team_name(nm or "")

    t_home = get_team_name(home)
    t_away = get_team_name(away)

    # time / venue
    start_str = e.get("date", "")
    try:
        start_dt = parse_dt(start_str)
        start_dt_local = start_dt.astimezone(tz.gettz(TIMEZONE))
        time_str = start_dt_local.strftime("%-I:%M %p")
        date_nice = start_dt_local.strftime("%b %-d, %Y")
    except Exception:
        time_str = ""
        date_nice = date.strftime("%b %-d, %Y")

    venue = (comp.get("venue", {}) or {}).get("fullName", "")
    games.append({
        "gameId": e.get("id"),
        "home": t_home, "away": t_away,
        "location": venue or "",
        "time_str": time_str,
        "date_str": date_nice
    })

games_df = pd.DataFrame(games)

# Pull BPI table
with st.spinner("Pulling ESPN BPI…"):
    bpi_df = fetch_espn_bpi_table(season_year=date.year if date.month>=11 else date.year-1)

# Optional fallback Torvik
if use_fallback:
    try:
        torvik_df = fetch_torvik_rating(year=date.year if date.month>=11 else date.year-1)
    except Exception:
        torvik_df = pd.DataFrame(columns=["Team","Team_norm","TRank"])

# Map team -> BPI
team_list = bpi_df["Team_norm"].tolist()
def lookup_bpi(team: str):
    # exact first
    row = bpi_df.loc[bpi_df["Team_norm"] == team]
    if not row.empty:
        return float(row.iloc[0]["BPI"]) if pd.notnull(row.iloc[0]["BPI"]) else None
    # fuzzy fallback
    cand = best_match(team, team_list)
    row = bpi_df.loc[bpi_df["Team_norm"] == cand]
    if not row.empty:
        return float(row.iloc[0]["BPI"]) if pd.notnull(row.iloc[0]["BPI"]) else None
    return None

out_rows = []
for r in games:
    h, a = r["home"], r["away"]
    bpi_h = lookup_bpi(h)
    bpi_a = lookup_bpi(a)
    diff = None
    if bpi_h is not None and bpi_a is not None:
        diff = bpi_h - bpi_a  # positive => home stronger
    row = {
        **r,
        "BPI_home": bpi_h,
        "BPI_away": bpi_a,
        "BPI_diff": diff,
    }
    if use_fallback:
        # TRank join (optional compare)
        def lookup_tr(team):
            if "Team_norm" in torvik_df.columns:
                z = torvik_df.loc[torvik_df["Team_norm"] == team]
                if not z.empty:
                    return float(z.iloc[0]["TRank"]) if pd.notnull(z.iloc[0]["TRank"]) else None
        row["TRank_home"] = lookup_tr(h)
        row["TRank_away"] = lookup_tr(a)
    out_rows.append(row)

merged = pd.DataFrame(out_rows)
merged = merged.sort_values(by=["BPI_diff"], ascending=False, na_position="last").reset_index(drop=True)

st.subheader("Matchups & Ratings (ESPN BPI)")
show_cols = ["time_str","away","BPI_away","home","BPI_home","BPI_diff","location"]
st.dataframe(merged[show_cols], use_container_width=True)

st.caption("BPI ≈ expected point margin per ~70 possessions vs. average team on neutral court (predictive power index). Source: ESPN BPI. Last updated daily. ")  #  College Basketball BPI** page (definitions + rendered table; updates daily).  [oai_citation:3‡ESPN.com](https://www.espn.com/mens-college-basketball/bpi)  
- Bart Torvik’s public CSV endpoints (fallback/compare), explicitly permitted to fetch (e.g., `getadvstats.php?year=YYYY&csv=1`).  [oai_citation:4‡adamcwisports.blogspot.com](https://adamcwisports.blogspot.com/p/data.html)

---

If you want, I can also add a toggle for **ESPN BPI game predictions** (win prob / projected margin) using ESPN’s predictions page as a secondary source, or add your brand colors/logo on the cards.
