import io
import zipfile
import math
from datetime import datetime
from dateutil import tz
from dateutil.parser import parse as parse_dt
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from rapidfuzz import process, fuzz
import streamlit as st

# =========================
# Page config / constants
# =========================
st.set_page_config(page_title="CBB Matchups • ESPN BPI", page_icon="🏀", layout="wide")
PRIMARY = "#0B1020"
LIGHT = "#E2E8F0"
TIMEZONE = "America/New_York"
DIV1_GROUP_ID = "50"  # ESPN uses 50 for Division I

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
    "Miami": "Miami (FL)",
}

def clean_team(name: str) -> str:
    if not name:
        return ""
    name = name.replace(" St.", " State").replace(" Univ.", " University").strip()
    return TEAM_ALIASES.get(name, name)

# =========================
# Safe formatter
# =========================
def fmt_signed(x):
    """Format as +1.1 or -1.1, return em dash for non-numeric."""
    try:
        v = float(x)
        if math.isnan(v):
            return "—"
        return f"{v:+.1f}"
    except Exception:
        return "—"

# =========================
# Helpers
# =========================
def best_match(name: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    m = process.extractOne(name, candidates, scorer=fuzz.WRatio, score_cutoff=80)
    return m[0] if m else None

def team_logo_from_competitor(comp) -> Optional[str]:
    try:
        t = comp.get("team", {}) if isinstance(comp, dict) else {}
    except Exception:
        return None
    if not isinstance(t, dict):
        return None
    if t.get("logo"):
        return t.get("logo")
    logos = t.get("logos", [])
    if isinstance(logos, list) and logos and isinstance(logos[0], dict):
        return logos[0].get("href")
    return None

def season_year_from_date(d: datetime.date) -> int:
    """Torvik typically uses the postseason year (e.g., Dec 2025 => 2026)."""
    return d.year + (1 if d.month >= 11 else 0)

# =========================
# Data fetch: scoreboard
# =========================
@st.cache_data(ttl=3*60*60, show_spinner=False)
def fetch_espn_scoreboard(date_yyyymmdd: str) -> dict:
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {"dates": date_yyyymmdd, "groups": DIV1_GROUP_ID, "limit": 500}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

# =========================
# Data fetch: ESPN BPI (robust)
# =========================
@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_via_api() -> Optional[pd.DataFrame]:
    urls = [
        "https://site.web.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
        "https://site.api.espn.com/apis/fitt/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
        "https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/rankings?type=bpi",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code != 200:
                continue
            data = r.json()
            items = None
            for key in ("items", "rankings", "data", "teams"):
                if isinstance(data, dict) and isinstance(data.get(key), list):
                    items = data[key]
                    break
            if not items:
                continue

            def get(d, *path, default=None):
                cur = d
                for p in path:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        return default
                return cur

            rows = []
            for it in items:
                nm = get(it, "team", "displayName") or get(it, "team", "name") or it.get("name")
                bpi = get(it, "metrics", "bpi", "value") or it.get("bpi") or get(it, "ratings", "bpi")
                if nm is None or bpi is None:
                    continue
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
        return [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns.to_list()]
    return [str(c) for c in df.columns]

def _score_table(df: pd.DataFrame) -> Tuple[int, Optional[str], bool]:
    cols = _flatten_columns(df)
    lower = [c.lower().strip() for c in cols]
    has_team = any("team" in c for c in lower)

    bpi_col = None
    for c in cols:
        cl = c.lower()
        if cl == "bpi" or cl.startswith("bpi "):
            bpi_col = c
            break

    num_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.3 * len(df))):
            num_cols.append(c)

    score = 0
    if bpi_col: score += 60
    if any(c.startswith("off") for c in lower): score += 10
    if any(c.startswith("def") for c in lower): score += 10
    if has_team: score += 10
    score += min(20, len(num_cols))
    score += min(20, len(df) // 25)
    return score, bpi_col, has_team

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi_via_html() -> pd.DataFrame:
    url = "https://www.espn.com/mens-college-basketball/bpi"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25).text

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

    if not candidates:
        soup = BeautifulSoup(html, "lxml")
        table_tag = soup.find("table")
        if table_tag is None:
            raise RuntimeError("Could not find any table on ESPN BPI page.")
        df = pd.read_html(str(table_tag))[0]
        df.columns = _flatten_columns(df)
        score, bpi_col, has_team = _score_table(df)
        candidates.append((score, df, bpi_col, has_team))

    candidates.sort(key=lambda x: x[0], reverse=True)
    cand, bpi_col, has_team = candidates[0][1], candidates[0][2], candidates[0][3]

    if not has_team:
        cand = cand.rename(columns={cand.columns[0]: "Team"})
    else:
        for c in list(cand.columns):
            if "team" in c.lower():
                cand = cand.rename(columns={c: "Team"})
                break

    if not bpi_col:
        num_cols = []
        for c in list(cand.columns):
            s = pd.to_numeric(cand[c], errors="coerce")
            if s.notna().sum() >= max(5, int(0.3 * len(cand))):
                med = float(s.median())
                num_cols.append((abs(med), c, med))
        good = [tpl for tpl in num_cols if -50 <= tpl[2] <= 50]
        pick = (sorted(good, key=lambda x: abs(x[2]), reverse=True) or num_cols or [(0, None, 0)])[0][1]
        bpi_col = pick

    if bpi_col is None or "Team" not in cand.columns:
        raise RuntimeError("Could not identify BPI table structure on ESPN.")

    df = cand.rename(columns={bpi_col: "BPI"})[["Team", "BPI"]].copy()
    df["Team"] = df["Team"].astype(str).str.replace(r"\s+\(\d+\)$", "", regex=True).str.strip()
    df["BPI"] = pd.to_numeric(df["BPI"], errors="coerce")
    df = df.dropna(subset=["Team", "BPI"])
    df["Team_norm"] = df["Team"].apply(clean_team)
    return df.drop_duplicates("Team_norm")[["Team", "Team_norm", "BPI"]]

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_torvik_rating(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(url, header=None)
        if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].lower().startswith("team"):
            df.columns = df.iloc[0]
            df = df[1:]
    if "Team" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Team"})
    df.columns = [str(c).strip() for c in df.columns]
    df["Team_norm"] = df["Team"].apply(clean_team)
    if "ADJEM" in df.columns:
        df["TRank"] = pd.to_numeric(df["ADJEM"], errors="coerce")
    else:
        alt = [c for c in df.columns if c.lower() in ("adjem", "adj_em", "rating", "margin")]
        df["TRank"] = pd.to_numeric(df[alt[0]], errors="coerce") if alt else pd.NA
    df = df.dropna(subset=["Team_norm"]).reset_index(drop=True)
    return df[["Team", "Team_norm", "TRank"]]

@st.cache_data(ttl=6*60*60, show_spinner=False)
def fetch_espn_bpi(season_year: int, emergency_csv_url: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str]:
    api_df = fetch_espn_bpi_via_api()
    if api_df is not None and len(api_df) > 0:
        return api_df, None, "ESPN API"

    try:
        html_df = fetch_espn_bpi_via_html()
        if len(html_df) > 0:
            return html_df, None, "ESPN HTML"
    except Exception:
        pass

    if emergency_csv_url:
        try:
            df = pd.read_csv(emergency_csv_url)
            cols = {c.lower().strip(): c for c in df.columns}
            team_col = cols.get("team") or cols.get("school") or list(df.columns)[0]
            rating_col = cols.get("bpi") or cols.get("adjem") or cols.get("rating") or list(df.columns)[1]
            out = pd.DataFrame({
                "Team": df[team_col].astype(str),
                "BPI": pd.to_numeric(df[rating_col], errors="coerce")
            })
            out["Team_norm"] = out["Team"].apply(clean_team)
            out = out.dropna(subset=["BPI"]).drop_duplicates("Team_norm")
            if not out.empty:
                return out[["Team", "Team_norm", "BPI"]], None, "CSV URL"
        except Exception:
            pass

    tdf = fetch_torvik_rating(season_year)
    out = tdf.rename(columns={"TRank": "BPI"}).dropna(subset=["BPI"]).copy()
    if not out.empty:
        return out[["Team", "Team_norm", "BPI"]], tdf, "Torvik fallback"

    return pd.DataFrame(columns=["Team", "Team_norm", "BPI"]), None, "None"

# =========================
# Fonts
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

TITLE_FONT = load_font(48)
MED_FONT = load_font(36)
BIG_FONT = load_font(96)
SMALL_FONT = load_font(28)
BADGE_FONT = load_font(72)

# =========================
# Logo fallback (badges)
# =========================
def _initials(team_name: str, max_letters: int = 3) -> str:
    if not team_name:
        return "—"
    parts = team_name.replace("-", " ").split()
    up = [p for p in parts if p.isupper() and len(p) <= 4]
    if up:
        s = "".join(up)[:max_letters]
        return s if s else team_name[:max_letters].upper()
    stop = {"of", "the", "and", "at", "st", "state"}
    letters = [w[0] for w in parts if w and w.lower() not in stop]
    if letters:
        return "".join(letters)[:max_letters].upper()
    return team_name[:max_letters].upper()

def draw_badge(team_name: str, size=(200, 200), color=(35, 48, 77)) -> Image.Image:
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, W, H), fill=color)
    txt = _initials(team_name)
    bbox = d.textbbox((0, 0), txt, font=BADGE_FONT)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((W - w) / 2, (H - h) / 2 - 4), txt, fill=(255, 255, 255), font=BADGE_FONT)
    return img

def fetch_logo_img(url: Optional[str], team_name: str, size=(200, 200)) -> Image.Image:
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return draw_badge(team_name, size=size)
    try:
        r = requests.get(url, timeout=10, stream=True)
        if r.status_code != 200:
            return draw_badge(team_name, size=size)
        ctype = r.headers.get("Content-Type", "")
        if "image" not in ctype:
            return draw_badge(team_name, size=size)
        content = r.content
        im = Image.open(io.BytesIO(content)).convert("RGBA")
        return im.resize(size)
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError):
        return draw_badge(team_name, size=size)

# =========================
# Card renderers
# =========================
def draw_edge_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, value: float, vmin=-15.0, vmax=15.0):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(24, 30, 54))
    v = max(min(value, vmax), vmin)
    pct = (v - vmin) / (vmax - vmin)
    fill_w = int(w * pct)
    col = (20, 184, 166) if value >= 0 else (249, 115, 22)
    draw.rounded_rectangle((x, y, x + fill_w, y + h), radius=8, fill=col)

def draw_card_original(row: pd.Series, brand_text: str = "CBB Edges", brand_logo_url: Optional[str] = None) -> Image.Image:
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), color=(11, 16, 32))
    d = ImageDraw.Draw(img)

    # Accent bands
    d.polygon([(0, int(H * 0.64)), (int(W * 0.62), H), (0, H)], fill=(20, 184, 166))
    d.polygon([(W, int(H * 0.34)), (W, H), (int(W * 0.38), H)], fill=(249, 115, 22))

    # Header
    hdr = f"{row.get('date_str', '')}  •  ESPN BPI"
    d.text((60, 40), hdr, fill=(248, 250, 252), font=MED_FONT)

    # Team blocks
    left, top = 80, 150
    away, home = str(row["away"]), str(row["home"])
    bpi_a, bpi_h = row.get("BPI_away"), row.get("BPI_home")
    diff = row.get("BPI_diff")

    d.text((left, top), away, fill=(255, 255, 255), font=TITLE_FONT)
    d.text((left, top + 80), f"BPI {fmt_signed(bpi_a)}", fill=(235, 240, 245), font=MED_FONT)

    d.text((left, top + 210), "@", fill=(200, 205, 220), font=MED_FONT)

    d.text((left, top + 300), home, fill=(255, 255, 255), font=TITLE_FONT)
    d.text((left, top + 380), f"BPI {fmt_signed(bpi_h)}", fill=(235, 240, 245), font=MED_FONT)

    # Center: diff number + bar
    center_x = int(W * 0.67)
    d.text((center_x, 160), "Rating Gap (Home − Away)", fill=(240, 242, 245), font=MED_FONT, anchor="lm")
    d.text((center_x, 235), fmt_signed(diff), fill=(255, 255, 255), font=BIG_FONT, anchor="lm")
    draw_edge_bar(d, center_x, 380, 500, 26, float(diff) if isinstance(diff, (float, int)) and not math.isnan(diff) else 0.0)

    # Logos/badges
    a_mark = fetch_logo_img(row.get("Away Logo"), away, size=(240, 240))
    h_mark = fetch_logo_img(row.get("Home Logo"), home, size=(240, 240))
    img.paste(a_mark, (120, 540), a_mark)
    img.paste(h_mark, (W - 120 - 240, 540), h_mark)

    # Footer
    footer = f"{row.get('time_str', '')} ET"
    loc = row.get("location", "")
    if loc:
        footer += f" • {loc}"
    d.text((60, H - 90), footer, fill=(16, 24, 40), font=TITLE_FONT)

    # Brand
    if brand_logo_url:
        bl = fetch_logo_img(brand_logo_url, "Brand", size=(160, 160))
        img.paste(bl, (W - 200, 40), bl)
    else:
        d.text((W - 60, 60), brand_text, fill=(230, 235, 240), font=MED_FONT, anchor="ra")
    return img

def draw_card_simple(row: pd.Series, brand_text: str = "CBB Edges") -> Image.Image:
    """
    Simple, table-like card: white background, two rows (Away, Home),
    columns: Logo | Team | BPI, with a top header showing Rating Gap and game info.
    """
    W, H = 1200, 675
    bg = (255, 255, 255)
    grid = (230, 230, 230)
    txt = (24, 24, 24)
    sub = (90, 90, 90)
    accent_pos = (20, 184, 166)
    accent_neg = (249, 115, 22)

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # Header
    header_h = 120
    d.rectangle((0, 0, W, header_h), fill=(250, 250, 250), outline=grid)
    title = f"{row.get('date_str','')}  •  ESPN BPI"
    d.text((24, 24), title, font=MED_FONT, fill=txt)
    meta = f"{row.get('time_str','')} ET"
    if row.get("location"):
        meta += f" • {row.get('location')}"
    d.text((24, 24 + 46), meta, font=SMALL_FONT, fill=sub)

    # Gap box (Home − Away)
    gap = row.get("BPI_diff")
    gap_txt = "—" if pd.isna(gap) else f"{gap:+.1f}"
    gap_color = accent_pos if (not pd.isna(gap) and gap >= 0) else accent_neg
    gap_w, gap_h = 220, 68
    gx, gy = W - gap_w - 24, 24
    d.rounded_rectangle((gx, gy, gx + gap_w, gy + gap_h), radius=10, fill=gap_color)
    d.text((gx + gap_w / 2, gy + gap_h / 2), f"Gap {gap_txt}", font=MED_FONT, fill=(255, 255, 255), anchor="mm")

    # Table area
    table_x, table_y = 24, header_h + 16
    table_w, table_h = W - 48, 420
    d.rectangle((table_x, table_y, table_x + table_w, table_y + table_h), outline=grid, fill=(255, 255, 255))

    # Column widths
    col_logo_w = 160
    col_team_w = 700
    col_bpi_w = table_w - col_logo_w - col_team_w

    # Header row
    row_h = 70
    d.rectangle((table_x, table_y, table_x + table_w, table_y + row_h), fill=(248, 248, 248), outline=grid)
    d.text((table_x + 16, table_y + row_h / 2), "Team", font=MED_FONT, fill=txt, anchor="lm")
    d.text((table_x + col_logo_w + col_team_w + 16, table_y + row_h / 2), "BPI", font=MED_FONT, fill=txt, anchor="lm")

    # Rows: Away then Home
    def draw_team_row(y, team, logo_url, bpi):
        d.line((table_x, y, table_x + table_w, y), fill=grid)
        mark = fetch_logo_img(logo_url, team, size=(120, 120))
        mx = table_x + 20
        my = int(y + (row_h * 2 - 120) / 2)
        img.paste(mark, (mx, my), mark)
        d.text((table_x + col_logo_w + 12, y + row_h), team, font=TITLE_FONT, fill=txt, anchor="lm")
        btxt = "—" if pd.isna(bpi) else f"{float(bpi):+.1f}"
        d.text((table_x + col_logo_w + col_team_w + 16, y + row_h), btxt, font=TITLE_FONT, fill=txt, anchor="lm")

    draw_team_row(table_y + row_h, str(row["away"]), row.get("Away Logo"), row.get("BPI_away"))
    draw_team_row(table_y + row_h * 3, str(row["home"]), row.get("Home Logo"), row.get("BPI_home"))

    # Footer brand
    d.line((24, H - 70, W - 24, H - 70), fill=grid)
    d.text((W - 24, H - 48), brand_text, font=SMALL_FONT, fill=sub, anchor="rs")

    return img

def draw_sheet_simple(rows: List[pd.Series], brand_text: str = "CBB Edges", cols: int = 2) -> Image.Image:
    cards = [draw_card_simple(r, brand_text=brand_text) for r in rows]
    if not cards:
        return Image.new("RGB", (1200, 600), (255, 255, 255))
    cw, ch = cards[0].size
    gap = 24
    col_count = max(1, int(cols))
    row_count = (len(cards) + col_count - 1) // col_count
    W = col_count * cw + (col_count + 1) * gap
    H = row_count * ch + (row_count + 1) * gap + 40
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 10), "CBB Matchups • ESPN BPI (Simple Sheet)", font=MED_FONT, fill=(24, 24, 24))
    for idx, card in enumerate(cards):
        r = idx // col_count
        c = idx % col_count
        x = gap + c * (cw + gap)
        y = gap + 30 + r * (ch + gap)
        sheet.paste(card, (x, y))
    d.text((W - gap, H - 20), brand_text, font=SMALL_FONT, fill=(100, 100, 100), anchor="rs")
    return sheet

# =========================
# UI header
# =========================
st.markdown(
    f"""
    <div style="padding:12px 16px;border-radius:12px;background:{PRIMARY};color:{LIGHT};margin-bottom:10px;">
      <div style="font-weight:800;font-size:22px;">🏀 College Hoops Matchups • ESPN BPI</div>
      <div style="opacity:.9;">Daily D-I games, BPI ratings, rating gaps, and exportable images (simple table or original style).</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Controls
# =========================
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    default_date = datetime.now(tz.gettz(TIMEZONE)).date()
    pick_date = st.date_input("Date", default_date)
with c2:
    max_cards = st.number_input("Max rows to export (by biggest gap)", min_value=1, max_value=200, value=24, step=1)
with c3:
    brand_text = st.text_input("Brand footer text", value="CLT Capper • CBB Edges")

brand_logo_url = st.text_input("Optional brand logo URL (used in Original style)", value="")
emergency_csv_url = st.text_input("Emergency ratings CSV URL (optional fallback)", value="")
debug_mode = st.checkbox("Debug: show data shapes/heads", value=False)

card_style = st.radio(
    "Card style",
    options=["Simple (table)", "Original"],
    index=0,
    horizontal=True
)

make_one_sheet = st.checkbox("Make one-sheet image (grid of simple cards)", value=False)
cols_per_sheet = st.slider("Columns on sheet (simple cards)", 1, 3, 2) if make_one_sheet else 2

date_str = pick_date.strftime("%Y%m%d")
season_year = season_year_from_date(pick_date)

# =========================
# Fetch games
# =========================
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
if debug_mode:
    st.write("Games shape:", games_df.shape)
    st.write(games_df.head())

# =========================
# Fetch ratings
# =========================
with st.spinner("Fetching ratings (ESPN BPI → HTML → Torvik → CSV)..."):
    bpi_df, torvik_df, bpi_source = fetch_espn_bpi(season_year, emergency_csv_url or None)

if debug_mode:
    st.write("Ratings source:", bpi_source)
    st.write("BPI df shape:", None if bpi_df is None else bpi_df.shape)
    if bpi_df is not None and not bpi_df.empty:
        st.write(bpi_df.head())
    if torvik_df is not None:
        st.write("Torvik df shape:", torvik_df.shape)
        st.write(torvik_df.head())

if bpi_df is None or bpi_df.empty:
    st.error("Could not load any ratings (BPI or fallback). If this persists, paste a CSV URL above temporarily.")
    st.stop()

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
    bpi_h = lookup_bpi(g["home"])
    bpi_a = lookup_bpi(g["away"])
    diff = (bpi_h - bpi_a) if (bpi_h is not None and bpi_a is not None) else None
    rows.append({**g.to_dict(), "BPI_home": bpi_h, "BPI_away": bpi_a, "BPI_diff": diff})

merged = pd.DataFrame(rows).sort_values(by=["BPI_diff"], ascending=False, na_position="last").reset_index(drop=True)

# =========================
# Display table (safe formatting)
# =========================
st.subheader("Matchups and Ratings")
if bpi_source == "Torvik fallback":
    st.info(f"ESPN BPI was unavailable; showing Torvik ADJEM (season {season_year}) as a fallback for rating gaps.")
else:
    st.caption(f"Source: {bpi_source}")

show_cols = ["time_str", "away", "BPI_away", "home", "BPI_home", "BPI_diff", "location"]

for c in ["BPI_home", "BPI_away", "BPI_diff"]:
    if c in merged.columns:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

def bg(val):
    try:
        v = float(val)
        if math.isnan(v):
            return ""
        if v >= 5:
            return "background-color: rgba(20,184,166,0.18);"
        if v <= -5:
            return "background-color: rgba(249,115,22,0.18);"
        return ""
    except Exception:
        return ""

fmt_map = {"BPI_home": fmt_signed, "BPI_away": fmt_signed, "BPI_diff": fmt_signed}

styled = (
    merged[show_cols]
      .style
      .format(fmt_map)
      .applymap(bg, subset=["BPI_diff"])
)
st.dataframe(styled, use_container_width=True)

# =========================
# Export images
# =========================
st.markdown("---")
st.subheader("Export images")

selection = st.multiselect(
    "Pick specific games (or leave blank to export the top N by gap):",
    options=list(merged.index),
    format_func=lambda i: f"{merged.loc[i,'away']} @ {merged.loc[i,'home']} • {merged.loc[i,'time_str']} ET"
)

to_export = merged.loc[selection] if selection else merged.head(int(max_cards))

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate ZIP of individual images"):
        images = []
        for _, row in to_export.iterrows():
            if pd.isna(row["BPI_home"]) or pd.isna(row["BPI_away"]):
                continue
            if card_style.startswith("Simple"):
                card = draw_card_simple(row, brand_text=brand_text)
            else:
                card = draw_card_original(row, brand_text=brand_text, brand_logo_url=brand_logo_url or None)
            fname = f"{row['away']} at {row['home']} - {row['date_str']}.png".replace("/", "-")
            buf = io.BytesIO()
            card.save(buf, format="PNG")
            images.append((fname, buf.getvalue()))

        if not images:
            st.warning("Nothing to export (ratings missing for selected games).")
        else:
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fname, data in images:
                    zf.writestr(fname, data)
            st.download_button(
                "Download ZIP",
                data=bio.getvalue(),
                file_name=f"cbb_cards_{pick_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
            )

with col2:
    if make_one_sheet and st.button("Generate one-sheet image"):
        rows_for_sheet = []
        for _, row in to_export.iterrows():
            if pd.isna(row["BPI_home"]) or pd.isna(row["BPI_away"]):
                continue
            rows_for_sheet.append(row)

        if not rows_for_sheet:
            st.warning("Nothing to include (ratings missing for selected games).")
        else:
            sheet = draw_sheet_simple(rows_for_sheet, brand_text=brand_text, cols=int(cols_per_sheet))
            bio = io.BytesIO()
            sheet.save(bio, format="PNG")
            st.download_button(
                "Download Sheet PNG",
                data=bio.getvalue(),
                file_name=f"cbb_sheet_{pick_date.strftime('%Y%m%d')}.png",
                mime="image/png",
            )

st.caption("BPI is ESPN's predictive power rating. Rating gap = Home minus Away (home − away).")
