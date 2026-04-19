from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import time

# ── Correct URLs from index page ─────────────────────────────────────
INVESTORS = {
    # ── Already in your list ──────────────────────────────────────────
    "Dolly Khanna":                       "https://trendlyne.com/portfolio/superstar-shareholders/53757/latest/dolly-khanna-portfolio/",
    "Ashish Kacholia":                    "https://trendlyne.com/portfolio/superstar-shareholders/53746/latest/ashish-kacholia-portfolio/",
    "Vijay Kedia":                        "https://trendlyne.com/portfolio/superstar-shareholders/53805/latest/vijay-kishanlal-kedia-portfolio/",
    "Rakesh Jhunjhunwala":               "https://trendlyne.com/portfolio/superstar-shareholders/53781/latest/rakesh-jhunjhunwala-and-associates-portfolio/",
    "Rekha Jhunjhunwala":                "https://trendlyne.com/portfolio/superstar-shareholders/53782/latest/rekha-jhunjhunwala-portfolio/",
    "Porinju Veliyath":                   "https://trendlyne.com/portfolio/superstar-shareholders/53777/latest/porinju-v-veliyath-portfolio/",
    "Sunil Singhania":                    "https://trendlyne.com/portfolio/superstar-shareholders/182955/latest/sunil-singhania-portfolio/",
    "Mukul Agrawal":                      "https://trendlyne.com/portfolio/superstar-shareholders/53774/latest/mukul-agrawal-portfolio/",
    "Radhakishan Damani":                "https://trendlyne.com/portfolio/superstar-shareholders/178317/latest/radhakishan-damani-portfolio/",
    "Madhusudan Kela":                    "https://trendlyne.com/portfolio/superstar-shareholders/584325/latest/madhusudan-kela-portfolio/",
    "Anil Kumar Goel":                    "https://trendlyne.com/portfolio/superstar-shareholders/53743/latest/anil-kumar-goel-and-associates-portfolio/",
    "Mohnish Pabrai":                     "https://trendlyne.com/portfolio/superstar-shareholders/69664/latest/mohnish-pabrai-portfolio/",
    "Nemish S Shah":                      "https://trendlyne.com/portfolio/superstar-shareholders/53776/latest/nemish-s-shah-portfolio/",
    "Akash Bhanshali":                    "https://trendlyne.com/portfolio/superstar-shareholders/53740/latest/akash-bhanshali-portfolio/",
    "Satpal Khattar":                     "https://trendlyne.com/portfolio/superstar-shareholders/53793/latest/satpal-khattar-portfolio/",

    # ── Additional investors (from Trendlyne index page) ─────────────
    "Ajay Upadhyaya":                     "https://trendlyne.com/portfolio/superstar-shareholders/53739/latest/ajay-upadhyaya-portfolio/",
    "Amit Gupta":                         "https://trendlyne.com/portfolio/superstar-shareholders/53741/latest/amit-gupta-portfolio/",
    "Anuj Anantrai Sheth":               "https://trendlyne.com/portfolio/superstar-shareholders/53744/latest/anuj-anantrai-sheth-and-associates-portfolio/",
    "Ashish Dhawan":                      "https://trendlyne.com/portfolio/superstar-shareholders/53745/latest/ashish-dhawan-portfolio/",
    "Ashok Kumar Jain":                   "https://trendlyne.com/portfolio/superstar-shareholders/53748/latest/ashok-kumar-jain-portfolio/",
    "Atim Kabra":                         "https://trendlyne.com/portfolio/superstar-shareholders/53749/latest/atim-kabra-portfolio/",
    "Bharat Jayantilal Patel":           "https://trendlyne.com/portfolio/superstar-shareholders/53751/latest/bharat-jayantilal-patel-and-associates-portfolio/",
    "Chandrakant Sampat":                "https://trendlyne.com/portfolio/superstar-shareholders/53752/latest/chandrakant-sampat-portfolio/",
    "Dilipkumar Lakhi":                   "https://trendlyne.com/portfolio/superstar-shareholders/53755/latest/dilipkumar-lakhi-portfolio/",
    "Girish Gulati":                      "https://trendlyne.com/portfolio/superstar-shareholders/53758/latest/girish-gulati-portfolio/",
    "Hitesh Satishchandra Doshi":        "https://trendlyne.com/portfolio/superstar-shareholders/53760/latest/hitesh-satishchandra-doshi-portfolio/",
    "Jagdish Master":                     "https://trendlyne.com/portfolio/superstar-shareholders/53761/latest/jagdish-master-portfolio/",
    "Ketan Mehta":                        "https://trendlyne.com/portfolio/superstar-shareholders/53763/latest/ketan-mehta-portfolio/",
    "Manish Jain":                        "https://trendlyne.com/portfolio/superstar-shareholders/53767/latest/manish-jain-portfolio/",
    "Mehul Madhusudan Shah":             "https://trendlyne.com/portfolio/superstar-shareholders/53770/latest/mehul-madhusudan-shah-portfolio/",
    "Mita Dipak Shah":                    "https://trendlyne.com/portfolio/superstar-shareholders/53771/latest/mita-dipak-shah-portfolio/",
    "Nalanda India Fund":                 "https://trendlyne.com/portfolio/superstar-shareholders/130252/latest/nalanda-india-fund-portfolio/",
    "Nikhil Vora":                        "https://trendlyne.com/portfolio/superstar-shareholders/53808/latest/nikhil-vora-portfolio/",
    "Pramod Bhasin":                      "https://trendlyne.com/portfolio/superstar-shareholders/53778/latest/pramod-bhasin-portfolio/",
    "Prashant Jain":                      "https://trendlyne.com/portfolio/superstar-shareholders/53779/latest/prashant-jain-portfolio/",
    "Pulak Prasad":                       "https://trendlyne.com/portfolio/superstar-shareholders/185406/latest/pulak-prasad-portfolio/",
    "Ramesh Damani":                      "https://trendlyne.com/portfolio/superstar-shareholders/53783/latest/ramesh-damani-portfolio/",
    "Ravi Dharamshi":                     "https://trendlyne.com/portfolio/superstar-shareholders/53784/latest/ravi-dharamshi-portfolio/",
    "S Naren":                            "https://trendlyne.com/portfolio/superstar-shareholders/53789/latest/s-naren-portfolio/",
    "Shankar Sharma":                     "https://trendlyne.com/portfolio/superstar-shareholders/53796/latest/shankar-sharma-portfolio/",
    "Shyam Sekhar":                       "https://trendlyne.com/portfolio/superstar-shareholders/53798/latest/shyam-sekhar-portfolio/",
    "Smallcap World Fund":               "https://trendlyne.com/portfolio/superstar-shareholders/53799/latest/smallcap-world-fund-portfolio/",
    "Sanjay Bakshi":                      "https://trendlyne.com/portfolio/superstar-shareholders/53790/latest/sanjay-bakshi-portfolio/",
    "Sanjiv Dhireshbhai Shah":           "https://trendlyne.com/portfolio/superstar-shareholders/53791/latest/sanjiv-dhireshbhai-shah-portfolio/",
    "Suresh Kumar Agrawal":              "https://trendlyne.com/portfolio/superstar-shareholders/53802/latest/suresh-kumar-agrawal-portfolio/",
    "Tulsian PMS":                        "https://trendlyne.com/portfolio/superstar-shareholders/53804/latest/tulsian-pms-portfolio/",
    "Uma Shanti Birla":                   "https://trendlyne.com/portfolio/superstar-shareholders/53806/latest/uma-shanti-birla-portfolio/",
    "Vijay Kishanlal Kedia (HUF)":       "https://trendlyne.com/portfolio/superstar-shareholders/53807/latest/vijay-kishanlal-kedia-huf-portfolio/",
    "Vinod Khanna":                       "https://trendlyne.com/portfolio/superstar-shareholders/53809/latest/vinod-khanna-portfolio/",
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