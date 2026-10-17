# ENSO Outlook: setup (no coding)
1. github.com → New repository → name `enso-dashboard` → Public → Create.
2. Add file → Upload files → drag in the unzipped folder contents (`docs`, `scripts`, `requirements.txt`). Commit.
3. Hidden `.github` folder: Add file → Create new file → type `.github/workflows/update.yml`, paste the file contents, Commit.
4. Settings → Actions → General → Workflow permissions → "Read and write" → Save.
5. Settings → Pages → Branch `main`, folder `/docs` → Save. Your URL: https://YOURNAME.github.io/enso-dashboard/
6. Actions tab → "Update ENSO forecast" → Run workflow. Wait ~2 min, refresh your URL.
7. Google Sites: Insert → Embed → By URL → paste your URL → set full width.
The page now includes an India monsoon (ENSO-tilt) table for the Northeast and Southwest monsoons.
It then refreshes itself every day.
