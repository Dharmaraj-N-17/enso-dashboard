"""Fetch NOAA CPC ENSO data, train one Ridge model per lead (1-6 months), write docs/forecast.json."""
import io, json, os, urllib.request
from datetime import datetime, timezone
from math import erf, sqrt
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://www.cpc.ncep.noaa.gov/data/indices/"
OUT = os.environ.get("OUT", "docs/forecast.json")
LEADS = range(1, 7)

def get(name):
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "enso-dashboard"})
    return urllib.request.urlopen(req, timeout=60).read().decode()

def load_sst():
    df = pd.read_csv(io.StringIO(get("sstoi.indices")), sep=r"\s+")
    df.columns = ["yr", "mon", "n12", "n12a", "n3", "n3a", "n4", "n4a", "n34", "n34a"]
    df.index = pd.to_datetime(dict(year=df.yr, month=df.mon, day=1))
    return df[["n3a", "n4a", "n34a"]].astype(float)

def load_soi():
    rows, on = {}, False
    for line in get("soi").splitlines():
        if "STANDARDIZED" in line:
            on = True
            continue
        p = line.split()
        if on and len(p) == 13 and p[0].isdigit():
            for m, v in enumerate(p[1:], 1):
                if float(v) > -99:
                    rows[pd.Timestamp(int(p[0]), m, 1)] = float(v)
    return pd.Series(rows, name="soi").sort_index()

def feats(d):
    f = pd.DataFrame(index=d.index)
    for k in range(4):
        f[f"n34_{k}"] = d.n34a.shift(k)
    f["n3"], f["n4"] = d.n3a, d.n4a
    f["soi0"], f["soi3"] = d.soi, d.soi.rolling(3).mean()
    f["ms"], f["mc"] = np.sin(2*np.pi*d.index.month/12), np.cos(2*np.pi*d.index.month/12)
    return f

def cdf(x, m, s):
    return 0.5 * (1 + erf((x - m) / (s * sqrt(2))))

def india(d, fc, last):
    """Rule-based ENSO tilt for Indian monsoon seasons (not a rainfall forecast)."""
    s = d.n34a.copy()
    for f in fc:
        s[pd.Timestamp(f["month"] + "-01")] = f["nino34"]
    def mk(name, region, months, year, el, la, ne):
        x = s[(s.index.year == year) & (s.index.month.isin(months))]
        if x.empty:
            return dict(name=name, region=region, window=f"{year}", basis="beyond forecast horizon", mean=None,
                        tilt="Too early. Watch Ni\u00f1o 3.4 through spring (Mar-May is the least predictable period).")
        m = float(x.mean()); nf = int((x.index > last).sum())
        basis = "observed" if nf == 0 else "forecast" if nf == len(x) else "part observed, part forecast"
        if len(x) < len(months): basis += f" ({len(x)} of {len(months)} months)"
        tilt = el if m >= 0.5 else la if m <= -0.5 else ne
        return dict(name=name, region=region, window=str(year), basis=basis, mean=round(m, 2), tilt=tilt)
    yo = last.year if last.month >= 7 else last.year - 1
    yj = last.year if last.month <= 9 else last.year + 1
    return [
        mk("Northeast monsoon (Oct-Dec)", "Tamil Nadu & south peninsula", [10, 11, 12], yo,
           "Tilts toward above-normal rain (El Ni\u00f1o tends to strengthen this season).",
           "Tilts toward below-normal rain (La Ni\u00f1a tends to weaken this season).",
           "No clear ENSO signal; normal year-to-year variability expected."),
        mk("Southwest monsoon (Jun-Sep)", "All India", [6, 7, 8, 9], yj,
           "Raised risk of below-normal rain, mainly central and northwest India. Not guaranteed.",
           "Tilts toward normal or above-normal rain.",
           "No clear ENSO signal; other factors will decide the season.")]

def main():
    d = load_sst().join(load_soi(), how="left")
    d = d[d.index >= "1951-01-01"].copy()
    d["soi"] = d["soi"].ffill(limit=2).fillna(0)  # SOI publishes a bit late
    X = feats(d)
    last = X.index[-1]
    mk = lambda: make_pipeline(StandardScaler(), Ridge(alpha=10))
    fc, skill = [], []
    for h in LEADS:
        y = d.n34a.shift(-h)
        ok = X.notna().all(axis=1) & y.notna()
        Xt, yt = X[ok], y[ok]
        pr, tr_, pe = [], [], []
        for a, b in TimeSeriesSplit(n_splits=5, gap=h).split(Xt):
            m = mk().fit(Xt.iloc[a], yt.iloc[a])
            pr += list(m.predict(Xt.iloc[b])); tr_ += list(yt.iloc[b]); pe += list(Xt.iloc[b].n34_0)
        pr, tr_, pe = map(np.array, (pr, tr_, pe))
        rm, rp = float(np.sqrt(np.mean((pr-tr_)**2))), float(np.sqrt(np.mean((pe-tr_)**2)))
        p = float(mk().fit(Xt, yt).predict(X.loc[[last]])[0])
        pel, pla = 1 - cdf(0.5, p, rm), cdf(-0.5, p, rm)
        fc.append(dict(lead=h, month=(last + pd.DateOffset(months=h)).strftime("%Y-%m"),
                       nino34=round(p, 2), lo80=round(p-1.28*rm, 2), hi80=round(p+1.28*rm, 2),
                       p_el=round(pel, 3), p_neutral=round(1-pel-pla, 3), p_la=round(pla, 3)))
        skill.append(dict(lead=h, rmse_model=round(rm, 3), rmse_persistence=round(rp, 3),
                          corr=round(float(np.corrcoef(pr, tr_)[0, 1]), 2),
                          improvement_pct=round(100*(1-rm/rp), 1)))
    ind = india(d, fc, last)
    cur = float(d.n34a.iloc[-1])
    hist = d.tail(240)
    out = dict(placeholder=False, updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               last_data_month=last.strftime("%Y-%m"), current_nino34=round(cur, 2),
               state="El Niño conditions" if cur >= .5 else "La Niña conditions" if cur <= -.5 else "Neutral",
               history=dict(month=[t.strftime("%Y-%m") for t in hist.index],
                            nino34=[round(v, 2) for v in hist.n34a], soi=[round(v, 2) for v in hist.soi]),
               forecast=fc, skill=skill, india=ind)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print("OK", out["last_data_month"], out["state"], [f["nino34"] for f in fc])

if __name__ == "__main__":
    main()
