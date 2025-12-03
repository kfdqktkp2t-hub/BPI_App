import io
import zipfile
from datetime import datetime, date as date_cls
from dateutil import tz
from dateutil.parser import parse as parse_dt
from typing import Dict, List, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import process, fuzz
import streamlit as st

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="CBB Matchups • ESPN BPI", page_icon="🏀", layout="wide")

# Basic colors (ASCII only to avoid encoding issues)
PRIMARY = "#0B1020"   # dark background
ACCENT = "#14B8A6"    # teal
ACCENT2 = "#F97316"   # orange
LIGHT = "#E2E8F0"
TIMEZONE = "America/New_York"
DIV1_GROUP_ID = "50"  # ESPN uses 50 for D-I

# =========================
# Team name normalization
# =========================
TEAM_ALIASES: Dict[str, str] = {
    "UConn": "Connecticut",
    "Ole Miss": "Mississippi",
    "UNC": "North Carolina",
    "USC": "Southern California",
    "UTSA": "UT San Antonio",
    "UC Santa Barbara": "Santa Barbara",
    "NC State": "North Carolina St.",
    "UMass": "Massachusetts",
    "UT Arlington": "Texas Arlington",
    "UT Rio Grande Valley": "Texas Rio Grande Valley",
    "Central Florida": "UCF",
    "Miami": "Miami (FL)",  # ESPN often uses (FL) for NCAAM
}

def clean_team(name: str) -> str:
    if not name:
        return ""
    name = name.replace(" St.", " State").replace(" Univ.", " University").strip()
    return TEAM_ALIASES.get(name, name)

# =========================
# Data fetchers (cached)
# =========================
@st.cache_data(ttl=3*60*60, show_spinner=False)
def fetch_espn_scoreboard(date_yyyymmdd: str) -> dict:
    """ESPN men's CBB scoreboard JSON (public endpoint)."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {"dates": date_yyyymmdd, "groups": DIV1_GROUP_ID, "limit": 500}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_via_api() -> Optional[pd.DataFrame]:
    """
    First attempt: ESPN web API for BPI rankings (undocumented; may change).
    If it works, it is cleaner than scraping HTML.
    """
    possible_urls = [
        # Try a couple of likely paths used across ESPN sports stacks
        "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
        "https://site.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
        "https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in possible_urls:
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code != 200:
                continue
            data = r.json()
            # Try to find a list of teams with bpi value
            items = None
            for key in ("items", "rankings", "data", "teams"):
                if isinstance(data, dict) and key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            if not items:
                continue
            rows = []
            def get(d, *path, default=None):
                cur = d
                for p in path:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        return default
                return cur
            for it in items:
                # Heuristic paths; ESPN JSON can be nested
                team_name = get(it, "team", "displayName") or get(it, "team", "name") or get(it, "name")
                bpi_val = get(it, "metrics", "bpi", "value")
                if bpi_val is None:
                    bpi_val = it.get("bpi") or get(it, "ratings", "bpi")
                if team_name is None or bpi_val is None:
                    continue
                rows.append({"Team": str(team_name).strip(), "BPI": float(bpi_val)})
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["Team_norm"] = df["Team"].apply(clean_team)
            df = df.drop_duplicates(subset=["Team_norm"]).reset_index(drop=True)
            return df[["Team", "Team_norm", "BPI"]]
        except Exception:
            continue
    return None

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_via_html() -> pd.DataFrame:
    """
    Fallback: scrape BPI table from the ESPN BPI page HTML using pandas.read_html.
    If ESPN changes markup, we attempt a BeautifulSoup fallback to the first table.
    """
    url = "https://www.espn.com/mens-college-basketball/bpi"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=25).text

    # Try all tables via pandas
    tables = []
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        tables = []

    candidate = None
    for t in tables:
        cols = [str(c).lower() for c in list(t.columns)]
        if any("team" in c for c in cols) and any("bpi" in c for c in cols):
            candidate = t.copy()
            break

    if candidate is None:
        # Soup fallback: first <table> on page
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if table is None:
            raise RuntimeError("Could not find a BPI table on ESPN page.")
        candidate = pd.read_html(str(table))[0]

    # Normalize columns
    candidate.columns = [str(c).strip() for c in candidate.columns]
    ren = {}
    for c in candidate.columns:
        cl = c.lower()
        if "team" in cl:
            ren[c] = "Team"
        elif cl == "bpi" or "bpi" in cl and "rk" not in cl:
            ren[c] = "BPI"
        elif "rk" in cl and "bpi" in cl:
            ren[c] = "BPI_RK"
    if ren:
        candidate = candidate.rename(columns=ren)

    if "Team" not in candidate.columns or "BPI" not in candidate.columns:
        # Best effort: pick first numeric column as BPI
        num_cols = [c for c in candidate.columns if pd.to_numeric(candidate[c], errors="coerce").notna().sum() > 0]
        if "Team" not in candidate.columns:
            candidate = candidate.rename(columns={candidate.columns[0]: "Team"})
        if "BPI" not in candidate.columns and num_cols:
            candidate = candidate.rename(columns={num_cols[0]: "BPI"})

    # Clean and keep
    candidate["Team"] = candidate["Team"].astype(str).str.replace(r"\s+\(\d+\)$", "", regex=True).str.strip()
    candidate["BPI"] = pd.to_numeric(candidate["BPI"], errors="coerce")
    candidate = candidate.dropna(subset=["Team", "BPI"]).copy()
    candidate["Team_norm"] = candidate["Team"].apply(clean_team)
    candidate = candidate.drop_duplicates(subset=["Team_norm"]).reset_index(drop=True)
    return candidate[["Team", "Team_norm", "BPI"]]

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi() -> pd.DataFrame:
    """Try API first, then HTML."""
    api_df = fetch_espn_bpi_via_api()
    if api_df is not None and len(api_df) > 0:
        return api_df
    return fetch_espn_bpi_via_html()

# =========================
# Helpers
# =========================
def local_time_from_iso(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = parse_dt(iso_str)
        local = dt.astimezone(tz.gettz(TIMEZONE))
        return local.strftime("%-I:%M %p")
    except Exception:
        return ""

def team_logo_from_competitor(comp) -> Optional[str]:
    t = comp.get("team", {}) if isinstance(comp, dict) else {}
    if t.get("logo"):
        return t.get("logo")
    logos = t.get("logos", [])
    if isinstance(logos, list) and logos:
        return logos[0].get("href")
    return None

def best_match(name: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    m = process.extractOne(name, candidates, scorer=fuzz.WRatio, score_cutoff=80)
    return m[0] if m else None

# =========================
# Image card rendering
# =========================
def load_font(size: int) -> ImageFont.FreeTypeFont:
    paths = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

TITLE_FONT = load_font(58)
MED_FONT = load_font(42)
BIG_FONT = load_font(112)
SMALL_FONT = load_font(28)

def fetch_logo_img(url: Optional[str], size=(260, 260)) -> Optional[Image.Image]:
    if not url:
        return None
    try:
        img = Image.open(io.BytesIO(requests.get(url, timeout=12).content)).convert("RGBA")
        return img.resize(size)
    except Exception:
        return None

def draw_edge_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, value: float, vmin=-15.0, vmax=15.0):
    # Background
    draw.rounded_rectangle((x, y, x+w, y+h), radius=8, fill=(24, 30, 54))
    # Map value to position
    v = max(min(value, vmax), vmin)
    pct = (v - vmin) / (vmax - vmin)
    fill_w = int(w * pct)
    col = (20, 184, 166) if value >= 0 else (249, 115, 22)
    draw.rounded_rectangle((x, y, x+fill_w, y+h), radius=8, fill=col)

def draw_card(row: pd.Series, brand_text: str = "CBB Edges", brand_logo_url: Optional[str] = None) -> Image.Image:
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), color=(11, 16, 32))
    d = ImageDraw.Draw(img)

    # Accent bands
    d.polygon([(0, int(H*0.64)), (int(W*0.62), H), (0, H)], fill=(20, 184, 166))
    d.polygon([(W, int(H*0.34)), (W, H), (int(W*0.38), H)], fill=(249, 115, 22))

    # Header
    hdr = f"{row.get('date_str','')}  •  ESPN BPI"
    d.text((60, 40), hdr, fill=(248, 250, 252), font=MED_FONT)

    # Team blocks
    left = 80
    top = 150
    away = str(row["away"])
    home = str(row["home"])
    bpi_a = row.get("BPI_away")
    bpi_h = row.get("BPI_home")
    diff = row.get("BPI_diff")

    d.text((left, top), away, fill=(255,255,255), font=TITLE_FONT)
    d.text((left, top+80), f"BPI {bpi_a:+.1f}" if pd.notna(bpi_a) else "BPI —", fill=(235,240,245), font=MED_FONT)

    d.text((left, top+210), "@", fill=(200,205,220), font=MED_FONT)

    d.text((left, top+300), home, fill=(255,255,255), font=TITLE_FONT)
    d.text((left, top+380), f"BPI {bpi_h:+.1f}" if pd.notna(bpi_h) else "BPI —", fill=(235,240,245), font=MED_FONT)

    # Center: diff number + bar
    center_x = int(W*0.67)
    d.text((center_x, 160), "Rating Gap (Home − Away)", fill=(240, 242, 245), font=MED_FONT, anchor="lm")
    diff_txt = f"{diff:+.1f}" if pd.notna(diff) else "—"
    d.text((center_x, 235), diff_txt, fill=(255,255,255), font=BIG_FONT, anchor="lm")

    # Edge bar
    draw_edge_bar(d, center_x, 380, 500, 26, float(diff) if pd.notna(diff) else 0.0)

    # Logos
    a_logo = fetch_logo_img(row.get("Away Logo"))
    h_logo = fetch_logo_img(row.get("Home Logo"))
    if a_logo:
        img.paste(a_logo, (120, 540), a_logo)
    if h_logo:
        img.paste(h_logo, (W-120-260, 540), h_logo)

    # Footer: time and location
    footer = f"{row.get('time_str','')} ET"
    loc = row.get("location", "")
    if loc:
        footer += f" • {loc}"
    d.text((60, H-90), footer, fill=(16,24,40), font=TITLE_FONT)

    # Brand stamp (text or logo)
    if brand_logo_url:
        bl = fetch_logo_img(brand_logo_url, size=(180,180))
        if bl:
            img.paste(bl, (W-220, 40), bl)
        else:
            d.text((W-60, 60), brand_text, fill=(230,235,240), font=MED_FONT, anchor="ra")
    else:
        d.text((W-60, 60), brand_text, fill=(230,235,240), font=MED_FONT, anchor="ra")

    return img

# =========================
# UI chrome
# =========================
st.markdown(
    f"""
    <div style="padding:12px 16px;border-radius:12px;background:{PRIMARY};color:{LIGHT};margin-bottom:10px;">
      <div style="font-weight:800;font-size:22px;">🏀 College Hoops Matchups • ESPN BPI</div>
      <div style="opacity:.9;">Daily D-I games, BPI ratings, rating gaps, and exportable social cards.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inputs
c1, c2, c3 = st.columns([1,1,1])
with c1:
    default_date = datetime.now(tz.gettz(TIMEZONE)).date()
    pick_date = st.date_input("Date", default_date)
with c2:
    max_cards = st.number_input("Max cards to export (by biggest gap)", min_value=1, max_value=200, value=24, step=1)
with c3:
    brand_text = st.text_input("Brand footer text", value="CLT Capper • CBB Edges")

brand_logo_url = st.text_input("Optional brand logo URL (PNG with transparent background works best)", value="")

# =========================
# Pull games and BPI
# =========================
date_str = pick_date.strftime("%Y%m%d")

with st.spinner("Fetching games..."):
    sb = fetch_espn_scoreboard(date_str)

events = sb.get("events", [])
if not events:
    st.info("No Division I games found for this date.")
    st.stop()

games = []
for ev in events:
    comp = ev.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        continue

    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

    def team_name(c):
        t = c.get("team", {}) if isinstance(c, dict) else {}
        nm = t.get("location") or t.get("shortDisplayName") or t.get("displayName") or t.get("name")
        return clean_team(nm or "")

    home_name = team_name(home)
    away_name = team_name(away)
    home_logo = team_logo_from_competitor(home)
    away_logo = team_logo_from_competitor(away)

    # time/local
    iso = ev.get("date")
    try:
        dt_ = parse_dt(iso)
        local = dt_.astimezone(tz.gettz(TIMEZONE))
        time_str = local.strftime("%-I:%M %p")
        nice_date = local.strftime("%b %-d, %Y")
    except Exception:
        time_str = ""
        nice_date = pick_date.strftime("%b %-d, %Y")

    venue = (comp.get("venue", {}) or {}).get("fullName", "")

    games.append({
        "gameId": ev.get("id"),
        "home": home_name,
        "away": away_name,
        "Home Logo": home_logo,
        "Away Logo": away_logo,
        "time_str": time_str,
        "date_str": nice_date,
        "location": venue,
    })

games_df = pd.DataFrame(games)

with st.spinner("Fetching ESPN BPI..."):
    bpi_df = fetch_espn_bpi()

# Map team -> BPI using exact then fuzzy
team_list = bpi_df["Team_norm"].tolist()

def lookup_bpi(team_norm: str) -> Optional[float]:
    exact = bpi_df.loc[bpi_df["Team_norm"] == team_norm]
    if not exact.empty:
        v = exact.iloc[0]["BPI"]
        return float(v) if pd.notna(v) else None
    m = best_match(team_norm, team_list)
    if not m:
        return None
    row = bpi_df.loc[bpi_df["Team_norm"] == m]
    if row.empty:
        return None
    v = row.iloc[0]["BPI"]
    return float(v) if pd.notna(v) else None

rows = []
for _, g in games_df.iterrows():
    h = g["home"]
    a = g["away"]
    bpi_h = lookup_bpi(h)
    bpi_a = lookup_bpi(a)
    diff = None
    if bpi_h is not None and bpi_a is not None:
        diff = bpi_h - bpi_a
    rows.append({
        **g.to_dict(),
        "BPI_home": bpi_h,
        "BPI_away": bpi_a,
        "BPI_diff": diff,
    })

merged = pd.DataFrame(rows).sort_values(by=["BPI_diff"], ascending=False, na_position="last").reset_index(drop=True)

# =========================
# Display table
# =========================
st.subheader("Matchups and Ratings (ESPN BPI)")
show_cols = ["time_str", "away", "BPI_away", "home", "BPI_home", "BPI_diff", "location"]
fmt = {c: "{:+.1f}" for c in ["BPI_home", "BPI_away", "BPI_diff"]}
def bg(val):
    try:
        if pd.isna(val):
            return ""
        if val >= 5:   # green-ish
            return "background-color: rgba(20,184,166,0.18);"
        if val <= -5:  # orange-ish
            return "background-color: rgba(249,115,22,0.18);"
        return ""
    except Exception:
        return ""

styled = (merged[show_cols].style
          .format(fmt)
          .applymap(bg, subset=["BPI_diff"]))
st.dataframe(styled, use_container_width=True)

# =========================
# Export image cards
# =========================
st.markdown("---")
st.subheader("Export X-ready image cards")

selection = st.multiselect(
    "Pick specific games (or leave blank to export the top N by gap):",
    options=list(merged.index),
    format_func=lambda i: f"{merged.loc[i,'away']} @ {merged.loc[i,'home']} • {merged.loc[i,'time_str']} ET"
)

to_export = merged.loc[selection] if selection else merged.head(int(max_cards))

if st.button("Generate and download ZIP"):
    images = []
    for _, row in to_export.iterrows():
        if pd.isna(row["BPI_home"]) or pd.isna(row["BPI_away"]):
            continue
        card = draw_card(row, brand_text=brand_text, brand_logo_url=brand_logo_url or None)
        fname = f"{row['away']} at {row['home']} - {row['date_str']}.png".replace("/", "-")
        images.append((fname, card))

    if not images:
        st.warning("Nothing to export (BPI missing for selected games).")
    else:
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, im in images:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                zf.writestr(fname, buf.getvalue())
        st.download_button(
            "Download ZIP of image cards",
            data=bio.getvalue(),
            file_name=f"cbb_bpi_cards_{pick_date.strftime('%Y%m%d')}.zip",
            mime="application/zip",
        )

# =========================
# Notes
# =========================
st.caption("BPI is ESPN's predictive power rating. Positive = above average; rating gap is Home minus Away.")
