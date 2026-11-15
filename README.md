# Equity Essence Scanner app

A phone app (installable web app) with end-of-day scans for **Large, Mid, Small and Micro caps**:

| Section | What it finds |
|---|---|
| Flag & Pole | Bull flag (weekly/monthly) with unusual volume at the pole top or in the flag |
| Elliott Wave 2 | First impulse off a major low: wave 2 / buy zone / wave 3 starting, with an unusual volume print |
| Base Breakout | 6+ month consolidation breaking out (or within 5%) with unusual volume |
| Stage 2 Early | Minervini Trend Template just turned true, or an early stage 2 advance |
| Stage 2 → 3 | Late stage 2 / topping, with volume drying up |
| Concalls | Latest earnings calls: results vs last year + key numbers from the transcript |

Every scan list keeps only companies that hold **regular concalls** (Screener.in: 4+ calls in 24 months, latest within 7 months).
Large = Nifty 100, Mid = rest of Nifty 500, Small = Nifty Smallcap 250, Micro = every other NSE stock with market cap >= Rs 500 cr.

## How it works
* `app_scan.py` runs every weekday at about 5 PM IST on GitHub (free), using the same scanner code as the laptop
  (`elliott_scanner.py`, `flag_pole_scanner.py`, `base_breakout_scanner.py`, `stage2_scanner.py`, `stage2_maturity.py`).
* It writes `docs/data/latest.json`. The app in `docs/` is served by GitHub Pages and reads that file.
* Concall key points come from the transcript PDF (no AI, no cost): sentences with numbers about guidance, orders,
  capex/capacity and margins, plus the quarter's results from Screener.

## One-time setup (about 15 minutes)
1. Create a free account at github.com.
2. Click **+ → New repository**. Name it `ee-scanner`, choose **Public** (GitHub Pages is free only for public
   repositories), tick nothing else, click **Create repository**.
3. On the new page click **uploading an existing file**. Open this `scanner_app` folder on your laptop, select
   **everything inside it** (including the `.github` folder — in File Explorer turn on View → Show → Hidden items
   if you don't see it) and drag it into the browser. Click **Commit changes**.
   If the `.github` folder did not upload, click **Add file → Create new file**, type the name
   `.github/workflows/eod.yml`, paste the contents of `eod.yml` from this folder, and commit.
4. **Settings → Actions → General → Workflow permissions →** choose **Read and write permissions** → Save.
5. **Settings → Pages →** Source: *Deploy from a branch*, Branch: `main`, folder `/docs` → Save.
   After a minute the page shows your app link: `https://<your-username>.github.io/ee-scanner/`.
6. **Actions tab → EOD scan → Run workflow** to run the first full scan now (takes 30–60 minutes). After that it
   runs by itself every weekday evening.

## Put it on your phone
* **Quick way:** open the app link in Chrome on Android → menu ⋮ → **Add to Home screen / Install app**.
  It opens full screen like any app and works offline with the last data.
* **Real APK file:** go to **pwabuilder.com**, paste the app link, click **Package for stores → Android →
  Generate**. Download the zip; the `.apk` inside installs on any Android phone (allow "install unknown apps"
  once). Free, no Android Studio needed.

## Notes
* The repository is public, so anyone with the link can see the scan lists.
* If Screener.in or BSE block GitHub's servers on some day, that day's concall checks fall back to the last saved
  data (`cache/`), and transcripts are retried the next day.
* Educational screening tool, not investment advice.
