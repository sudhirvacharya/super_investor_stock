"""
╔══════════════════════════════════════════════════════════════════════╗
║        SUPERSTAR INVESTOR PORTFOLIO SCRAPER — FULL INTELLIGENCE      ║
║  Features: auto-discovery, conviction scoring, buy/sell signals,     ║
║  top-10 picker, investor clustering, retry logic, HTML dashboard     ║
║  Output: <date><time>_Report.html  e.g. 20Apr2026_1433_Report.html  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import time
import random
import json
import os
from datetime import datetime

# ── Optional: clustering (pip install scikit-learn) ───────────────────
try:
    from sklearn.metrics.pairwise import cosine_similarity
    CLUSTERING_ENABLED = True
except ImportError:
    CLUSTERING_ENABLED = False
    print("[WARN] scikit-learn not installed. Clustering disabled.")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
INDEX_URL   = "https://trendlyne.com/portfolio/superstar-shareholders/index/individual/"
OUTPUT_JSON = "superstar_raw.json"
MIN_DELAY   = 2.5
MAX_DELAY   = 4.5
RETRIES     = 3


# ══════════════════════════════════════════════════════════════════════
# TIER 1 — AUTO-DISCOVER ALL INVESTOR URLS FROM INDEX PAGE
# ══════════════════════════════════════════════════════════════════════
def scrape_investor_index(page):
    """Scrape the Trendlyne index page to auto-build investor URL dict."""
    print("\n[STEP 1] Auto-discovering investor list from index page...")
    safe_goto(page, INDEX_URL)
    soup = BeautifulSoup(page.content(), "html.parser")
    investors = {}
    for a in soup.select("table a[href*='superstar-shareholders']"):
        href = a.get("href", "")
        name = a.get_text(strip=True)
        if name and "/latest/" in href:
            full_url = "https://trendlyne.com" + href if href.startswith("/") else href
            investors[name] = full_url
    print(f"  -> Found {len(investors)} investors")
    return investors


# ══════════════════════════════════════════════════════════════════════
# TIER 6 — RETRY + RATE LIMIT SAFE NAVIGATION
# ══════════════════════════════════════════════════════════════════════
def safe_goto(page, url, retries=RETRIES):
    """Navigate with retry + exponential backoff + human-like delay."""
    for attempt in range(retries):
        try:
            page.goto(url, wait_until="networkidle", timeout=35000)
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            return True
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  [RETRY {attempt+1}/{retries}] {e} — waiting {wait}s")
            time.sleep(wait)
    print(f"  [FAIL] Could not load: {url}")
    return False


# ══════════════════════════════════════════════════════════════════════
# TIER 2 — RICH PER-STOCK DATA SCRAPER
# ══════════════════════════════════════════════════════════════════════
def get_portfolio(page, url, name):
    """
    Scrape full portfolio: stock name, value, % portfolio, buy/sell signal.
    Returns list of dicts.
    """
    if not safe_goto(page, url):
        return []

    soup   = BeautifulSoup(page.content(), "html.parser")
    rows   = soup.select("table tbody tr")
    stocks = []

    for row in rows:
        cols = row.select("td")
        if len(cols) < 3:
            continue

        name_td    = row.select_one("td.stockName") or cols[0]
        stock_name = name_td.get_text(strip=True)
        if not stock_name or stock_name.lower() in ("stock", "company", ""):
            continue

        value_cr   = cols[1].get_text(strip=True) if len(cols) > 1 else ""
        pct_port   = cols[2].get_text(strip=True) if len(cols) > 2 else ""
        qty_change = cols[3].get_text(strip=True) if len(cols) > 3 else ""
        sector     = cols[4].get_text(strip=True) if len(cols) > 4 else ""

        if qty_change.startswith("+") or qty_change.lower() == "new":
            signal = "BUY"
        elif qty_change.startswith("-"):
            signal = "SELL"
        else:
            signal = "HOLD"

        stocks.append({
            "stock":      stock_name,
            "value_cr":   value_cr,
            "pct_port":   pct_port,
            "qty_change": qty_change,
            "signal":     signal,
            "sector":     sector,
        })

    return stocks


# ══════════════════════════════════════════════════════════════════════
# TIER 3 — SMART ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def build_master_df(all_holdings):
    rows = []
    for investor, stocks in all_holdings.items():
        for s in stocks:
            rows.append({"investor": investor, **s})
    return pd.DataFrame(rows)


def analyze(df):
    print("\n[STEP 3] Running analysis...")

    # ── Conviction Score ──────────────────────────────────────────────
    conviction = (
        df.groupby("stock")["investor"]
        .nunique()
        .reset_index()
        .rename(columns={"investor": "superstar_count"})
        .sort_values("superstar_count", ascending=False)
        .reset_index(drop=True)
    )
    conviction["investors_list"] = conviction["stock"].map(
        lambda s: ", ".join(df[df["stock"] == s]["investor"].unique().tolist())
    )

    # ── Fresh Buy Signal ──────────────────────────────────────────────
    buys_df  = df[df["signal"] == "BUY"]
    new_buys = (
        buys_df.groupby("stock")["investor"]
        .apply(lambda x: ", ".join(x.unique()))
        .reset_index()
        .rename(columns={"investor": "buyers"})
    )
    new_buys["buyer_count"] = new_buys["buyers"].apply(lambda x: len(x.split(", ")))
    new_buys = new_buys.sort_values("buyer_count", ascending=False).reset_index(drop=True)

    # ── Exit Signal ───────────────────────────────────────────────────
    sells_df = df[df["signal"] == "SELL"]
    exits = (
        sells_df.groupby("stock")["investor"]
        .apply(lambda x: ", ".join(x.unique()))
        .reset_index()
        .rename(columns={"investor": "sellers"})
    )
    exits["seller_count"] = exits["sellers"].apply(lambda x: len(x.split(", ")))
    exits = exits.sort_values("seller_count", ascending=False).reset_index(drop=True)

    # ── Sector Preference ─────────────────────────────────────────────
    if "sector" in df.columns and df["sector"].str.strip().any():
        sector_pref = (
            df[df["sector"].str.strip() != ""]
            .groupby("sector")["stock"]
            .count()
            .reset_index()
            .rename(columns={"stock": "stock_count"})
            .sort_values("stock_count", ascending=False)
        )
    else:
        sector_pref = pd.DataFrame(columns=["sector", "stock_count"])

    return conviction, new_buys, exits, sector_pref


# ══════════════════════════════════════════════════════════════════════
# TIER 3b — TOP 10 PICKER
# ══════════════════════════════════════════════════════════════════════
def pick_top10(conviction, new_buys, exits, df):
    """
    Pick 10 stocks from conviction_count > 3.
    Priority order:
      1. Highest conviction count  (weight x4)
      2. Fresh BUY signal          (weight x2)
      3. Sector diversification    (max 3 per sector)
      4. Avoid recent SELLs        (penalty x1)
    """
    base = conviction[conviction["superstar_count"] > 3].copy()

    if base.empty:
        print("[WARN] No stocks with conviction > 3 found.")
        return pd.DataFrame()

    buy_map    = new_buys.set_index("stock")["buyer_count"].to_dict()
    sell_map   = exits.set_index("stock")["seller_count"].to_dict()
    sector_map = (
        df.groupby("stock")["sector"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        .to_dict()
    )

    base["buyer_count"]  = base["stock"].map(buy_map).fillna(0)
    base["seller_count"] = base["stock"].map(sell_map).fillna(0)
    base["sector"]       = base["stock"].map(sector_map).fillna("Unknown")

    # Composite score
    base["score"] = (
        base["superstar_count"] * 4   # priority 1 — conviction
      + base["buyer_count"]     * 2   # priority 2 — fresh buys
      - base["seller_count"]    * 1   # priority 4 — soft sell penalty
    )
    base = base.sort_values("score", ascending=False).reset_index(drop=True)

    # Sector diversification guard — max 3 stocks per sector
    selected, sector_counts = [], defaultdict(int)
    for _, row in base.iterrows():
        sec = row["sector"]
        if sector_counts[sec] < 3:
            selected.append(row)
            sector_counts[sec] += 1
        if len(selected) == 10:
            break

    cols = ["stock", "superstar_count", "buyer_count", "seller_count", "sector", "score", "investors_list"]
    return pd.DataFrame(selected)[cols]


# ══════════════════════════════════════════════════════════════════════
# TIER 4 — INVESTOR CLUSTERING (who thinks alike)
# ══════════════════════════════════════════════════════════════════════
def investor_similarity(df):
    if not CLUSTERING_ENABLED:
        return pd.DataFrame()

    print("\n[STEP 4] Computing investor similarity clusters...")
    pivot = (
        df.assign(held=1)
        .pivot_table(index="investor", columns="stock", values="held", fill_value=0)
    )
    sim_matrix = cosine_similarity(pivot)
    sim_df     = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)

    rows = []
    for investor in sim_df.index:
        top3 = sim_df[investor].drop(investor).nlargest(3)
        for peer, score in top3.items():
            rows.append({
                "investor":       investor,
                "similar_to":     peer,
                "similarity_pct": round(score * 100, 1),
            })
    result = pd.DataFrame(rows).sort_values(
        ["investor", "similarity_pct"], ascending=[True, False]
    )
    return result


# ══════════════════════════════════════════════════════════════════════
# TIER 5 — HTML DASHBOARD EXPORT
# ══════════════════════════════════════════════════════════════════════
def export_html(conviction, new_buys, exits, sector_pref, similarity_df, raw_df, top10_df, filename):
    import html as html_lib

    def df_to_html_rows(df, signal_col=None, highlight_col=None):
        rows_html = ""
        for _, row in df.iterrows():
            cells = ""
            for col in df.columns:
                val         = str(row[col]) if row[col] is not None else ""
                val_escaped = html_lib.escape(val)
                cls         = ""
                if signal_col and col == signal_col:
                    if val == "BUY":    cls = ' class="sig-buy"'
                    elif val == "SELL": cls = ' class="sig-sell"'
                    else:               cls = ' class="sig-hold"'
                elif highlight_col and col == highlight_col:
                    try:
                        n = float(val)
                        if n >= 5:   cls = ' class="conv-high"'
                        elif n >= 3: cls = ' class="conv-mid"'
                    except:
                        pass
                cells += f'<td{cls}>{val_escaped}</td>'
            rows_html += f"<tr>{cells}</tr>\n"
        return rows_html

    def make_table(tab_id, df, signal_col=None, highlight_col=None):
        headers = "".join(
            f'<th onclick="sortTable(this)">{html_lib.escape(c)} <span class="sort-icon">⇅</span></th>'
            for c in df.columns
        )
        rows = df_to_html_rows(df, signal_col=signal_col, highlight_col=highlight_col)
        return f"""
        <div class="tab-content" id="tab-{tab_id}">
          <div class="search-bar">
            <input type="text" placeholder="🔍 Filter rows..." oninput="filterTable(this, '{tab_id}-tbl')">
          </div>
          <div class="table-wrap">
            <table id="{tab_id}-tbl" class="data-table">
              <thead><tr>{headers}</tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </div>"""

    tabs = [
        ("top10",      "⭐ Top 10 Picks", top10_df,      None,     "score"),
        ("conviction", "🏆 Conviction",   conviction,    None,     "superstar_count"),
        ("buys",       "🟢 Fresh Buys",   new_buys,      None,     "buyer_count"),
        ("exits",      "🔴 Exits",        exits,         None,     "seller_count"),
        ("sectors",    "📊 Sectors",      sector_pref,   None,     None),
        ("clusters",   "🔗 Clusters",     similarity_df, None,     "similarity_pct"),
        ("raw",        "📋 Raw Data",     raw_df,        "signal", None),
    ]

    tab_buttons  = ""
    tab_contents = ""
    first = True
    for tid, label, df_tab, sig_col, hl_col in tabs:
        if df_tab is None or df_tab.empty:
            continue
        active        = "active" if first else ""
        tab_buttons  += (
            f'<button class="tab-btn {active}" onclick="showTab(\'{tid}\')" id="btn-{tid}">'
            f'{label} <span class="badge">{len(df_tab)}</span></button>\n'
        )
        content = make_table(tid, df_tab, signal_col=sig_col, highlight_col=hl_col)
        if not first:
            content = content.replace(f'id="tab-{tid}"', f'id="tab-{tid}" style="display:none"')
        tab_contents += content
        first = False

    now_str = datetime.now().strftime('%d %b %Y %H:%M')

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Superstar Investor Dashboard — {now_str}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  :root {{
    --bg:#0a0c10; --surface:#111318; --border:#1e2230;
    --accent:#f0b429; --accent2:#38bdf8;
    --green:#22c55e; --red:#ef4444;
    --text:#e2e8f0; --muted:#64748b;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'DM Mono',monospace; font-size:13px; }}
  .header {{ padding:32px 40px 24px; border-bottom:1px solid var(--border); background:linear-gradient(135deg,#0a0c10,#0f1520); }}
  .header h1 {{ font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:var(--accent); }}
  .header p {{ color:var(--muted); margin-top:4px; font-size:12px; }}
  .header-meta {{ display:flex; gap:12px; margin-top:16px; flex-wrap:wrap; }}
  .meta-pill {{ background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:6px 14px; font-size:11px; color:var(--accent2); }}
  .meta-pill span {{ color:var(--muted); margin-right:6px; }}
  .tabs {{ display:flex; gap:2px; padding:16px 40px 0; background:var(--surface); border-bottom:1px solid var(--border); overflow-x:auto; }}
  .tab-btn {{ background:none; border:none; color:var(--muted); font-family:'DM Mono',monospace; font-size:12px; padding:10px 18px; cursor:pointer; border-bottom:2px solid transparent; transition:all 0.2s; white-space:nowrap; }}
  .tab-btn:hover {{ color:var(--text); }}
  .tab-btn.active {{ color:var(--accent); border-bottom-color:var(--accent); }}
  .badge {{ background:var(--border); border-radius:10px; padding:1px 7px; font-size:10px; margin-left:4px; color:var(--muted); }}
  .tab-btn.active .badge {{ background:var(--accent); color:#000; }}
  .search-bar {{ padding:16px 40px 8px; }}
  .search-bar input {{ background:var(--surface); border:1px solid var(--border); border-radius:6px; color:var(--text); font-family:'DM Mono',monospace; font-size:12px; padding:8px 14px; width:300px; outline:none; transition:border-color 0.2s; }}
  .search-bar input:focus {{ border-color:var(--accent); }}
  .table-wrap {{ padding:0 40px 40px; overflow-x:auto; }}
  .data-table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  .data-table thead tr {{ background:var(--surface); border-bottom:2px solid var(--accent); }}
  .data-table th {{ padding:10px 14px; text-align:left; font-family:'Syne',sans-serif; font-weight:600; font-size:11px; letter-spacing:0.5px; color:var(--accent); cursor:pointer; white-space:nowrap; user-select:none; }}
  .data-table th:hover {{ color:var(--accent2); }}
  .sort-icon {{ opacity:0.4; font-size:10px; }}
  .data-table td {{ padding:9px 14px; border-bottom:1px solid var(--border); color:var(--text); max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .data-table tr:hover td {{ background:#ffffff08; }}
  .sig-buy  {{ color:var(--green) !important; font-weight:500; }}
  .sig-sell {{ color:var(--red)   !important; font-weight:500; }}
  .sig-hold {{ color:var(--muted) !important; }}
  .conv-high {{ background:#fbbf2420; color:var(--accent)  !important; font-weight:500; }}
  .conv-mid  {{ background:#38bdf810; color:var(--accent2) !important; }}
  .footer {{ text-align:center; padding:20px; color:var(--muted); font-size:11px; border-top:1px solid var(--border); }}
  ::-webkit-scrollbar {{ height:4px; width:4px; }}
  ::-webkit-scrollbar-track {{ background:var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:2px; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ Superstar Investor Dashboard</h1>
  <p>Portfolio intelligence across India's top superstar investors — sourced from Trendlyne</p>
  <div class="header-meta">
    <div class="meta-pill"><span>Investors</span>{raw_df['investor'].nunique()}</div>
    <div class="meta-pill"><span>Unique Stocks</span>{raw_df['stock'].nunique()}</div>
    <div class="meta-pill"><span>Fresh Buys</span>{len(new_buys)}</div>
    <div class="meta-pill"><span>Exits</span>{len(exits)}</div>
    <div class="meta-pill"><span>Generated</span>{now_str}</div>
    <div class="meta-pill"><span>File</span>{filename}</div>
  </div>
</div>
<div class="tabs">
{tab_buttons}
</div>
{tab_contents}
<div class="footer">Data sourced from Trendlyne.com · For informational purposes only · Not investment advice</div>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display='none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-'+id).style.display='block';
  document.getElementById('btn-'+id).classList.add('active');
}}
function filterTable(input, tableId) {{
  const q = input.value.toLowerCase();
  document.getElementById(tableId).querySelectorAll('tbody tr').forEach(row => {{
    row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function sortTable(th) {{
  const tbody = th.closest('table').querySelector('tbody');
  const idx   = Array.from(th.parentNode.children).indexOf(th);
  const asc   = th.dataset.asc !== 'true';
  th.dataset.asc = asc;
  Array.from(tbody.querySelectorAll('tr'))
    .sort((a,b) => {{
      const av = a.cells[idx]?.textContent.trim()||'';
      const bv = b.cells[idx]?.textContent.trim()||'';
      const an = parseFloat(av.replace(/[^0-9.-]/g,''));
      const bn = parseFloat(bv.replace(/[^0-9.-]/g,''));
      if (!isNaN(an)&&!isNaN(bn)) return asc ? an-bn : bn-an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }})
    .forEach(r => tbody.appendChild(r));
  th.querySelector('.sort-icon').textContent = asc ? ' ↑' : ' ↓';
}}
</script>
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"  -> Saved: {filename}  (open in browser)")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    start_time  = datetime.now()
    output_html = start_time.strftime("%d%b%Y_%H%M") + "_Report.html"

    print("=" * 60)
    print("  SUPERSTAR INVESTOR SCRAPER")
    print(f"  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output  : {output_html}")
    print("=" * 60)

    all_holdings = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

        # ── Step 1: Auto-discover investors ───────────────────────────
        investors = scrape_investor_index(page)

        # Fallback hardcoded list if index scrape fails
        if not investors:
            print("[WARN] Index scrape returned 0 investors. Using fallback list.")
            investors = {
                "Dolly Khanna":        "https://trendlyne.com/portfolio/superstar-shareholders/53757/latest/dolly-khanna-portfolio/",
                "Ashish Kacholia":     "https://trendlyne.com/portfolio/superstar-shareholders/53746/latest/ashish-kacholia-portfolio/",
                "Vijay Kedia":         "https://trendlyne.com/portfolio/superstar-shareholders/53805/latest/vijay-kishanlal-kedia-portfolio/",
                "Rakesh Jhunjhunwala": "https://trendlyne.com/portfolio/superstar-shareholders/53781/latest/rakesh-jhunjhunwala-and-associates-portfolio/",
                "Rekha Jhunjhunwala":  "https://trendlyne.com/portfolio/superstar-shareholders/53782/latest/rekha-jhunjhunwala-portfolio/",
                "Porinju Veliyath":    "https://trendlyne.com/portfolio/superstar-shareholders/53777/latest/porinju-v-veliyath-portfolio/",
                "Sunil Singhania":     "https://trendlyne.com/portfolio/superstar-shareholders/182955/latest/sunil-singhania-portfolio/",
                "Mukul Agrawal":       "https://trendlyne.com/portfolio/superstar-shareholders/53774/latest/mukul-agrawal-portfolio/",
                "Radhakishan Damani":  "https://trendlyne.com/portfolio/superstar-shareholders/178317/latest/radhakishan-damani-portfolio/",
                "Madhusudan Kela":     "https://trendlyne.com/portfolio/superstar-shareholders/584325/latest/madhusudan-kela-portfolio/",
                "Anil Kumar Goel":     "https://trendlyne.com/portfolio/superstar-shareholders/53743/latest/anil-kumar-goel-and-associates-portfolio/",
                "Mohnish Pabrai":      "https://trendlyne.com/portfolio/superstar-shareholders/69664/latest/mohnish-pabrai-portfolio/",
                "Nemish S Shah":       "https://trendlyne.com/portfolio/superstar-shareholders/53776/latest/nemish-s-shah-portfolio/",
                "Akash Bhanshali":     "https://trendlyne.com/portfolio/superstar-shareholders/53740/latest/akash-bhanshali-portfolio/",
                "Satpal Khattar":      "https://trendlyne.com/portfolio/superstar-shareholders/53793/latest/satpal-khattar-portfolio/",
            }

        # ── Step 2: Scrape each investor's portfolio ───────────────────
        print(f"\n[STEP 2] Scraping {len(investors)} investor portfolios...")
        for i, (investor, url) in enumerate(investors.items(), 1):
            print(f"\n  [{i}/{len(investors)}] {investor}")
            stocks = get_portfolio(page, url, investor)
            all_holdings[investor] = stocks
            print(f"    -> {len(stocks)} stocks | "
                  f"Buys: {sum(1 for s in stocks if s['signal']=='BUY')} | "
                  f"Sells: {sum(1 for s in stocks if s['signal']=='SELL')}")

        browser.close()

    # ── Save raw JSON ──────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_holdings, f, indent=2)
    print(f"\n  -> Raw data saved: {OUTPUT_JSON}")

    # ── Build master DataFrame ─────────────────────────────────────────
    df = build_master_df(all_holdings)
    if df.empty:
        print("[ERROR] No data scraped. Check selectors or site structure.")
        return

    # ── Analysis ───────────────────────────────────────────────────────
    conviction, new_buys, exits, sector_pref = analyze(df)

    # ── Top 10 Picker ──────────────────────────────────────────────────
    top10 = pick_top10(conviction, new_buys, exits, df)

    # ── Print results to console ───────────────────────────────────────
    print("\n── TOP 10 PICKS (scored) ──")
    if not top10.empty:
        print(top10.to_string(index=False))

    print("\n── TOP 20 CONVICTION STOCKS ──")
    print(conviction.head(20).to_string(index=False))

    print("\n── TOP 10 FRESH BUYS ──")
    print(new_buys.head(10).to_string(index=False))

    print("\n── TOP 10 EXITS ──")
    print(exits.head(10).to_string(index=False))

    # ── Clustering ─────────────────────────────────────────────────────
    similarity_df = investor_similarity(df)
    if not similarity_df.empty:
        print("\n── INVESTOR SIMILARITY (top pairs) ──")
        print(similarity_df.head(15).to_string(index=False))

    # ── Export HTML dashboard ──────────────────────────────────────────
    export_html(conviction, new_buys, exits, sector_pref, similarity_df, df, top10, filename=output_html)

    elapsed = (datetime.now() - start_time).seconds
    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed}s | "
          f"{len(investors)} investors | "
          f"{df['stock'].nunique()} unique stocks")
    print(f"  Report  : {output_html}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ── Optional: load from JSON cache to skip re-scraping ────────────
    # Set USE_CACHE = True to reload from superstar_raw.json
    # and regenerate the HTML without scraping again.
    USE_CACHE = False
    if USE_CACHE and os.path.exists(OUTPUT_JSON):
        print(f"[CACHE] Loading from {OUTPUT_JSON}...")
        output_html = datetime.now().strftime("%d%b%Y_%H%M") + "_Report.html"
        with open(OUTPUT_JSON) as f:
            all_holdings = json.load(f)
        df            = build_master_df(all_holdings)
        conviction, new_buys, exits, sector_pref = analyze(df)
        top10         = pick_top10(conviction, new_buys, exits, df)
        similarity_df = investor_similarity(df)
        export_html(conviction, new_buys, exits, sector_pref, similarity_df, df, top10, filename=output_html)
        print(f"  -> Report saved: {output_html}")
    else:
        main()