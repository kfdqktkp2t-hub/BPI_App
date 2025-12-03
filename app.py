import io, zipfile
from datetime import datetime
from dateutil import tz
from dateutil.parser import parse as parse_dt
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import process, fuzz
import streamlit as st

# -------------------- Page config / constants --------------------
st.set_page_config(page_title="CBB Matchups • ESPN BPI", page_icon="🏀", layout="wide")
PRIMARY = "#0B1020"; ACCENT = "#14B8A6"; ACCENT2 = "#F97316"; LIGHT = "#E2E8F0"
TIMEZONE = "America/New_York"; DIV1_GROUP_ID = "50"

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
    "Miami": "Miami (FL)",
}

def clean_team(name: str) -> str:
    if not name: return ""
    name = name.replace(" St.", " State").replace(" Univ.", " University").strip()
    return TEAM_ALIASES.get(name, name)

# -------------------- Helpers --------------------
def local_time_from_iso(iso_str: str) -> str:
    if not iso_str: return ""
    try:
        dt = parse_dt(iso_str)
        return dt.astimezone(tz.gettz(TIMEZONE)).strftime("%-I:%M %p")
    except Exception:
        return ""

def best_match(name: str, candidates: List[str]) -> Optional[str]:
    if not candidates: return None
    m = process.extractOne(name, candidates, scorer=fuzz.WRatio, score_cutoff=80)
    return m[0] if m else None

def team_logo_from_competitor(comp) -> Optional[str]:
    t = comp.get("team", {}) if isinstance(comp, dict) else {}
    if t.get("logo"): return t.get("logo")
    logos = t.get("logos", [])
    if isinstance(logos, list) and logos: return logos[0].get("href")
    return None

# -------------------- Data fetch: scoreboard --------------------
@st.cache_data(ttl=3*60*60, show_spinner=False)
def fetch_espn_scoreboard(date_yyyymmdd: str) -> dict:
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {"dates": date_yyyymmdd, "groups": DIV1_GROUP_ID, "limit": 500}
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
    return r.json()

# -------------------- Data fetch: ESPN BPI (robust) --------------------
@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_via_api() -> Optional[pd.DataFrame]:
    """Try a few public web API variants ESPN uses internally."""
    urls = [
        "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
        "https://site.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
        "https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code != 200: continue
            data = r.json()
            items = None
            for key in ("items", "rankings", "data", "teams"):
                if isinstance(data, dict) and isinstance(data.get(key), list):
                    items = data[key]; break
            if not items: continue

            rows = []
            def get(d, *path, default=None):
                cur = d
                for p in path:
                    if isinstance(cur, dict) and p in cur: cur = cur[p]
                    else: return default
                return cur
            for it in items:
                nm = get(it, "team", "displayName") or get(it, "team", "name") or it.get("name")
                bpi = get(it, "metrics", "bpi", "value") or it.get("bpi") or get(it, "ratings", "bpi")
                if nm is None or bpi is None: continue
                rows.append({"Team": str(nm).strip(), "BPI": float(bpi)})
            if rows:
                df = pd.DataFrame(rows)
                df["Team_norm"] = df["Team"].apply(clean_team)
                return df.drop_duplicates("Team_norm")[["Team", "Team_norm", "BPI"]]
        except Exception:
            continue
    return None

def _flatten_columns(df: pd.DataFrame) -> List[str]:
    if isinstance(df.columns, pd.MultiIndex):
        cols = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns.to_list()]
    else:
        cols = [str(c) for c in df.columns]
    return cols

def _score_table(df: pd.DataFrame) -> Tuple[int, Optional[str], bool]:
    """
    Score a candidate table: higher is more likely to be the BPI table.
    Returns (score, bpi_col_name, has_team_col)
    """
    cols = _flatten_columns(df)
    lower = [c.lower().strip() for c in cols]
    has_team = any("team" in c for c in lower)

    # identify BPI column if present
    bpi_col = None
    for c in cols:
        cl = c.lower()
        if cl == "bpi" or cl.startswith("bpi "):
            bpi_col = c; break

    # numeric columns
    num_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.3*len(df))):
            num_cols.append(c)

    # heuristic score
    score = 0
    if bpi_col: score += 60
    if any(c.startswith("off") for c in lower): score += 10
    if any(c.startswith("def") for c in lower): score += 10
    if has_team: score += 10
    score += min(20, len(num_cols))  # more numeric columns is good
    score += min(20, len(df)//25)    # more rows is good

    return score, bpi_col, has_team

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_via_html() -> pd.DataFrame:
    """
    Robust HTML parse: try all tables, pick the best-scoring candidate.
    Never raises KeyError; returns a DataFrame with Team, Team_norm, BPI.
    """
    url = "https://www.espn.com/mens-college-basketball/bpi"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25).text
    # First attempt: pandas.read_html finds many tables; we'll score them.
    tables = []
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        tables = []

    candidates: List[Tuple[int, pd.DataFrame, Optional[str], bool]] = []
    for t in tables:
        df = t.copy()
        df.columns = _flatten_columns(df)
        score, bpi_col, has_team = _score_table(df)
        candidates.append((score, df, bpi_col, has_team))

    # If nothing decent, look for the first <table> via BeautifulSoup as a fallback
    if not candidates:
        soup = BeautifulSoup(html, "lxml")
        table_tag = soup.find("table")
        if table_tag is None:
            raise RuntimeError("Could not find any table on ESPN BPI page.")
        df = pd.read_html(str(table_tag))[0]
        df.columns = _flatten_columns(df)
        score, bpi_col, has_team = _score_table(df)
        candidates.append((score, df, bpi_col, has_team))

    # Pick the top-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    cand, bpi_col, has_team = candidates[0][1], candidates[0][2], candidates[0][3]

    # Ensure Team column
    if not has_team:
        cand = cand.rename(columns={cand.columns[0]: "Team"})
    else:
        # rename whichever column contains 'team' to 'Team'
        for c in list(cand.columns):
            if "team" in c.lower():
                cand = cand.rename(columns={c: "Team"})
                break

    # Ensure BPI column
    if not bpi_col:
        # pick a numeric column whose median falls within a typical BPI range
        num_cols = []
        for c in list(cand.columns):
            s = pd.to_numeric(cand[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.3*len(cand))):
                med = float(s.median())
                num_cols.append((abs(med), c, med))
        # prefer columns with medians in [-50, 50]
        good = [tpl for tpl in num_cols if -50 <= tpl[2] <= 50]
        pick = (sorted(good, key=lambda x: abs(x[2]), reverse=True) or num_cols or [(0, None, 0)])[0][1]
        bpi_col = pick

    if bpi_col is None or "Team" not in cand.columns:
        raise RuntimeError("Could not identify BPI table structure on ESPN.")

    # Clean + return
    df = cand.rename(columns={bpi_col: "BPI"})[["Team", "BPI"]].copy()
    df["Team"] = df["Team"].astype(str).str.replace(r"\s+\(\d+\)$", "", regex=True).str.strip()
    df["BPI"] = pd.to_numeric(df["BPI"], errors="coerce")
    df = df.dropna(subset=["Team", "BPI"])
    df["Team_norm"] = df["Team"].apply(clean_team)
    return df.drop_duplicates("Team_norm")[["Team", "Team_norm", "BPI"]]

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_torvik_rating(year: int) -> pd.DataFrame:
    """Fallback: Torvik ADJEM (public CSV)."""
    url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
    df = pd.read_csv(url, header=None)
    if isinstance(df.iloc[0,0], str) and df.iloc[0,0].lower().startswith("team"):
        df.columns = df.iloc[0]; df = df[1:]
    if "Team" not in df.columns: df = df.rename(columns={df.columns[0]: "Team"})
    df["Team_norm"] = df["Team"].apply(clean_team)
    if "ADJEM" in df.columns:
        df["TRank"] = pd.to_numeric(df["ADJEM"], errors="coerce")
    else:
        df["TRank"] = pd.NA
    return df[["Team", "Team_norm", "TRank"]].dropna(subset=["Team_norm"]).reset_index(drop=True)

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Returns (bpi_df, torvik_df_if_used, source_label)
    source_label: "ESPN API", "ESPN HTML", or "Torvik fallback"
    """
    api_df = fetch_espn_bpi_via_api()
    if api_df is not None and len(api_df) > 0:
        return api_df, None, "ESPN API"
    try:
        html_df = fetch_espn_bpi_via_html()
        if len(html_df) > 0:
            return html_df, None, "ESPN HTML"
    except Exception:
        pass
    # last resort: Torvik
    yr = datetime.now().year if datetime.now().month >= 11 else datetime.now().year - 1
    tdf = fetch_torvik_rating(yr)
    # Map TRank -> BPI-like just for continuity (display will note fallback)
    out = tdf.rename(columns={"TRank": "BPI"}).dropna(subset=["BPI"]).copy()
    return out[["Team", "Team_norm", "BPI"]], tdf, "Torvik fallback"

# -------------------- Image card rendering --------------------
def load_font(size: int) -> ImageFont.FreeTypeFont:
    paths = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for p in paths:
        try: return ImageFont.truetype(p, size=size)
        except Exception: continue
    return ImageFont.load_default()

TITLE_FONT = load_font(58); MED_FONT = load_font(42); BIG_FONT = load_font(112)

def fetch_logo_img(url: Optional[str], size=(260, 260)) -> Optional[Image.Image]:
    if not url: return None
    try:
        img = Image.open(io.BytesIO(requests.get(url, timeout=12).content)).convert("RGBA")
        return img.resize(size)
    except Exception:
        return None

def draw_edge_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, value: float, vmin=-15.0, vmax=15.0):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=8, fill=(24, 30, 54))
    v = max(min(value, vmax), vmin); pct = (v - vmin) / (vmax - vmin)
    fill_w = int(w * pct); col = (20, 184, 166) if value >= 0 else (249, 115, 22)
    draw.rounded_rectangle((x, y, x+fill_w, y+h), radius=8, fill=col)

def draw_card(row: pd.Series, brand_text: str = "CBB Edges", brand_logo_url: Optional[str] = None) -> Image.Image:
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), color=(11, 16, 32)); d = ImageDraw.Draw(img)
    d.polygon([(0, int(H*0.64)), (int(W*0.62), H), (0, H)], fill=(20,184,166))
    d.polygon([(W, int(H*0.34)), (W, H), (int(W*0.38), H)], fill=(249,115,22))

    hdr = f"{row.get('date_str','')}  •  ESPN BPI"
    d.text((60, 40), hdr, fill=(248,250,252), font=MED_FONT)

    left, top = 80, 150
    away, home = str(row["away"]), str(row["home"])
    bpi_a, bpi_h, diff = row.get("BPI_away"), row.get("BPI_home"), row.get("BPI_diff")

    d.text((left, top), away, fill=(255,255,255), font=TITLE_FONT)
    d.text((left, top+80), f"BPI {bpi_a:+.1f}" if pd.notna(bpi_a) else "BPI —", fill=(235,240,245), font=MED_FONT)
    d.text((left, top+210), "@", fill=(200,205,220), font=MED_FONT)
    d.text((left, top+300), home, fill=(255,255,255), font=TITLE_FONT)
    d.text((left, top+380), f"BPI {bpi_h:+.1f}" if pd.notna(bpi_h) else "BPI —", fill=(235,240,245), font=MED_FONT)

    center_x = int(W*0.67)
    d.text((center_x, 160), "Rating Gap (Home − Away)", fill=(240,242,245), font=MED_FONT, anchor="lm")
    diff_txt = f"{diff:+.1f}" if pd.notna(diff) else "—"
    d.text((center_x, 235), diff_txt, fill=(255,255,255), font=BIG_FONT, anchor="lm")
    draw_edge_bar(d, center_x, 380, 500, 26, float(diff) if pd.notna(diff) else 0.0)

    a_logo = fetch_logo_img(row.get("Away Logo")); h_logo = fetch_logo_img(row.get("Home Logo"))
    if a_logo: img.paste(a_logo, (120, 540), a_logo)
    if h_logo: img.paste(h_logo, (W-120-260, 540), h_logo)

    footer = f"{row.get('time_str','')} ET"; loc = row.get("location", "")
    if loc: footer += f" • {loc}"
    d.text((60, H-90), footer, fill=(16,24,40), font=TITLE_FONT)

    if brand_logo_url:
        bl = fetch_logo_img(brand_logo_url, size=(180,180))
        if bl: img.paste(bl, (W-220, 40), bl)
        else:  d.text((W-60, 60), brand_text, fill=(230,235,240), font=MED_FONT, anchor="ra")
    else:
        d.text((W-60, 60), brand_text, fill=(230,235,240), font=MED_FONT, anchor="ra")
    return img

# -------------------- UI --------------------
st.markdown(
    f"""
    <div style="padding:12px 16px;border-radius:12px;background:{PRIMARY};color:{LIGHT};margin-bottom:10px;">
      <div style="font-weight:800;font-size:22px;">🏀 College Hoops Matchups • ESPN BPI</div>
      <div style="opacity:.9;">Daily D-I games, BPI ratings, rating gaps, and exportable social cards.</div>
    </div>
    """, unsafe_allow_html=True
)

c1, c2, c3 = st.columns([1,1,1])
with c1:
    default_date = datetime.now(tz.gettz(TIMEZONE)).date()
    pick_date = st.date_input("Date", default_date)
with c2:
    max_cards = st.number_input("Max cards to export (by biggest gap)", min_value=1, max_value=200, value=24, step=1)
with c3:
    brand_text = st.text_input("Brand footer text", value="CLT Capper • CBB Edges")
brand_logo_url = st.text_input("Optional brand logo URL (PNG w/ transparency works best)", value="")

date_str = pick_date.strftime("%Y%m%d")

# -------------------- Games --------------------
with st.spinner("Fetching games..."):
    sb = fetch_espn_scoreboard(date_str)
events = sb.get("events", [])
if not events:
    st.info("No Division I games found for this date."); st.stop()

games = []
for ev in events:
    comp = ev.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    if len(competitors) != 2: continue
    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

    def team_name(c):
        t = c.get("team", {}) if isinstance(c, dict) else {}
        nm = t.get("location") or t.get("shortDisplayName") or t.get("displayName") or t.get("name")
        return clean_team(nm or "")

    home_name, away_name = team_name(home), team_name(away)
    home_logo, away_logo = team_logo_from_competitor(home), team_logo_from_competitor(away)

    iso = ev.get("date")
    try:
        dt_ = parse_dt(iso); local = dt_.astimezone(tz.gettz(TIMEZONE))
        time_str = local.strftime("%-I:%M %p"); nice_date = local.strftime("%b %-d, %Y")
    except Exception:
        time_str = ""; nice_date = pick_date.strftime("%b %-d, %Y")
    venue = (comp.get("venue", {}) or {}).get("fullName", "")

    games.append({
        "gameId": ev.get("id"),
        "home": home_name, "away": away_name,
        "Home Logo": home_logo, "Away Logo": away_logo,
        "time_str": time_str, "date_str": nice_date, "location": venue,
    })
games_df = pd.DataFrame(games)

# -------------------- Ratings --------------------
with st.spinner("Fetching ESPN BPI..."):
    bpi_df, torvik_df, bpi_source = fetch_espn_bpi()

team_list = bpi_df["Team_norm"].tolist()

def lookup_bpi(team_norm: str) -> Optional[float]:
    exact = bpi_df.loc[bpi_df["Team_norm"] == team_norm]
    if not exact.empty:
        v = exact.iloc[0]["BPI"]; return float(v) if pd.notna(v) else None
    m = best_match(team_norm, team_list)
    if not m: return None
    row = bpi_df.loc[bpi_df["Team_norm"] == m]
    if row.empty: return None
    v = row.iloc[0]["BPI"]; return float(v) if pd.notna(v) else None

rows = []
for _, g in games_df.iterrows():
    bpi_h = lookup_bpi(g["home"]); bpi_a = lookup_bpi(g["away"])
    diff = (bpi_h - bpi_a) if (bpi_h is not None and bpi_a is not None) else None
    rows.append({**g.to_dict(), "BPI_home": bpi_h, "BPI_away": bpi_a, "BPI_diff": diff})
merged = pd.DataFrame(rows).sort_values(by=["BPI_diff"], ascending=False, na_position="last").reset_index(drop=True)

# -------------------- Table --------------------
st.subheader("Matchups and Ratings")
if bpi_source == "Torvik fallback":
    st.info("ESPN BPI was temporarily unavailable; showing Torvik ADJEM as a fallback for rating gaps.")
else:
    st.caption(f"Source: {bpi_source}")

show_cols = ["time_str", "away", "BPI_away", "home", "BPI_home", "BPI_diff", "location"]
fmt = {c: "{:+.1f}" for c in ["BPI_home", "BPI_away", "BPI_diff"]}

def bg(val):
    try:
        if pd.isna(val): return ""
        if val >= 5:  return "background-color: rgba(20,184,166,0.18);"
        if val <= -5: return "background-color: rgba(249,115,22,0.18);"
        return ""
    except Exception:
        return ""

styled = merged[show_cols].style.format(fmt).applymap(bg, subset=["BPI_diff"])
st.dataframe(styled, use_container_width=True)

# -------------------- Export image cards --------------------
st.markdown("---")
st.subheader("Export X-ready image cards")
selection = st.multiselect(
    "Pick specific games (or leave blank to export the top N by gap):",
    options=list(merged.index),
    format_func=lambda i: f"{merged.loc[i,'away']} @ {merged.loc[i,'home']} • {merged.loc[i,'time_str']} ET"
)
max_cards = int(max_cards)
to_export = merged.loc[selection] if selection else merged.head(max_cards)

if st.button("Generate and download ZIP"):
    images = []
    for _, row in to_export.iterrows():
        if pd.isna(row["BPI_home"]) or pd.isna(row["BPI_away"]): continue
        card = draw_card(row, brand_text=brand_text, brand_logo_url=brand_logo_url or None)
        fname = f"{row['away']} at {row['home']} - {row['date_str']}.png".replace("/", "-")
        images.append((fname, card))
    if not images:
        st.warning("Nothing to export (ratings missing for selected games).")
    else:
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, im in images:
                buf = io.BytesIO(); im.save(buf, format="PNG")
                zf.writestr(fname, buf.getvalue())
        st.download_button(
            "Download ZIP of image cards",
            data=bio.getvalue(),
            file_name=f"cbb_cards_{pick_date.strftime('%Y%m%d')}.zip",
            mime="application/zip",
        )

st.caption("BPI is ESPN's predictive power rating. Rating gap = Home minus Away.")
