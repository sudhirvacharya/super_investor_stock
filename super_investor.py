from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import time

# ── Correct URLs from index page ─────────────────────────────────────
INVESTORS = {
    "Dolly Khanna":               "https://trendlyne.com/portfolio/superstar-shareholders/53757/latest/dolly-khanna-portfolio/",
    "Ashish Kacholia":            "https://trendlyne.com/portfolio/superstar-shareholders/53746/latest/ashish-kacholia-portfolio/",
    "Vijay Kedia":                "https://trendlyne.com/portfolio/superstar-shareholders/53805/latest/vijay-kishanlal-kedia-portfolio/",
    "Rakesh Jhunjhunwala":        "https://trendlyne.com/portfolio/superstar-shareholders/53781/latest/rakesh-jhunjhunwala-and-associates-portfolio/",
    "Rekha Jhunjhunwala":         "https://trendlyne.com/portfolio/superstar-shareholders/53782/latest/rekha-jhunjhunwala-portfolio/",
    "Porinju Veliyath":           "https://trendlyne.com/portfolio/superstar-shareholders/53777/latest/porinju-v-veliyath-portfolio/",
    "Sunil Singhania":            "https://trendlyne.com/portfolio/superstar-shareholders/182955/latest/sunil-singhania-portfolio/",
    "Mukul Agrawal":              "https://trendlyne.com/portfolio/superstar-shareholders/53774/latest/mukul-agrawal-portfolio/",
    "Radhakishan Damani":         "https://trendlyne.com/portfolio/superstar-shareholders/178317/latest/radhakishan-damani-portfolio/",
    "Madhusudan Kela":            "https://trendlyne.com/portfolio/superstar-shareholders/584325/latest/madhusudan-kela-portfolio/",
    "Anil Kumar Goel":            "https://trendlyne.com/portfolio/superstar-shareholders/53743/latest/anil-kumar-goel-and-associates-portfolio/",
    "Mohnish Pabrai":             "https://trendlyne.com/portfolio/superstar-shareholders/69664/latest/mohnish-pabrai-portfolio/",
    "Nemish S Shah":              "https://trendlyne.com/portfolio/superstar-shareholders/53776/latest/nemish-s-shah-portfolio/",
    "Akash Bhanshali":            "https://trendlyne.com/portfolio/superstar-shareholders/53740/latest/akash-bhanshali-portfolio/",
    "Satpal Khattar":             "https://trendlyne.com/portfolio/superstar-shareholders/53793/latest/satpal-khattar-portfolio/",
}

def get_portfolio(page, url, name):
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(3)
        soup = BeautifulSoup(page.content(), "html.parser")
        stocks = []
        for td in soup.select("td.stockName"):
            stock = td.get_text(strip=True)
            if stock:
                stocks.append(stock)
        return stocks
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

all_holdings = {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_extra_http_headers({"User-Agent": "Mozilla/5.0"})

    for investor, url in INVESTORS.items():
        print(f"\nFetching: {investor}")
        stocks = get_portfolio(page, url, investor)
        all_holdings[investor] = stocks
        print(f"  -> {len(stocks)} stocks: {stocks[:3]}")
        time.sleep(2)

    browser.close()

# ── Find common stocks ────────────────────────────────────────────────
stock_count = Counter()
for stocks in all_holdings.values():
    stock_count.update(set(stocks))

print("\n── Top 20 common stocks across Super Investors ──")
df = pd.DataFrame(stock_count.most_common(20), columns=["Stock", "Investor Count"])
print(df.to_string(index=False))
df.to_csv("super_investor_common_stocks.csv", index=False)
print("\nSaved → super_investor_common_stocks.csv")