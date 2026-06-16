# Global renewable energy storage requirements from GLDAS

Python port and **global extension** of the cumulative surplus/deficit analysis in:

> Fekete BM, Bacskó M, Zhang J and Chen M (2023), *Storage requirements to mitigate
> intermittent renewable energy sources: analysis for the US Northeast.*
> Front. Environ. Sci. 11:1076830. doi:10.3389/fenvs.2023.1076830

Instead of NLDAS-2 daily means over CONUS, this pipeline uses **GLDAS Noah LSM
L4 3-hourly 0.25° V2.1 (`GLDAS_NOAH025_3H` 2.1)** forcings at full **3-hourly
resolution**, 2000–2025, to estimate — for every country, and the
states/provinces of large countries — the excess installation factor and the
energy storage capacity (as a fraction of annual consumption) needed for an
energy system relying solely on solar, wind, or their storage-optimal mix.

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 0. Regions (once) | `scripts/00_regions.py` | Natural Earth admin-0/admin-1 (bundled in `data/naturalearth/`) + one sample granule → `regions.gpkg/csv`, sparse cos(lat) weight matrix, land mask |
| 1. Zonal reduction | `scripts/01_zonal.py` | 3-hourly granules → 3-hourly area-weighted zonal means of `SWdown, Wind, Tair, Psurf, Qair` per region; one parquet per month. **No gridded data is stored.** |
| 2. Analysis | `scripts/02_analyze.py` | capacity factors + all analyses → `results/summary.csv` (per region) and `results/pooled_summary.csv` (continents + world) |
| 3. Figures | `scripts/03_figures.py` | paginated variability and deficit/storage PDFs (paper Figs. 5–12 analogues, daily-aggregated for legibility) + world choropleth maps |

## Methods

**From the paper / `Rscript/nldas_analys/Deficit.R`** (verified to machine
precision against an R-port reference in `tests/test_deficit_vs_r.py`):
capacity factor with starting/plateauing thresholds (Eq. 1), log-profile hub
height correction (Eqs. 2–3), excess installation factor (Eq. 4), net and
loss-adjusted cumulative deficits with k_R/k_D/k_aS efficiencies and the
iterative f_adj (Eqs. 5–9), and the storage-operation simulation. The
formulation is time-step agnostic (dt = 1/2920 for 3-hourly).

**Added in this work** (`gldas_storage/metrics.py`, `energy.py`):

1. **3-hourly resolution** — capacity factors and deficits computed at 3-hourly
   resolution capture the diurnal cycle the paper's daily means averaged away.
   The analysis also reports the **diurnal vs seasonal storage split**:
   S_seasonal from daily-mean CFs, S_diurnal = S_tot(3 h) − S_seasonal.
2. **Air-density-corrected wind power** — moist air density
   ρ = P/(R_d·T_v), T_v = T(1+0.608q) (Wallace & Hobbs 2006), surface pressure
   barometrically adjusted to hub height, and the IEC 61400-12-1 equivalent
   wind speed u_eq = u_hub(ρ/ρ₀)^(1/3), ρ₀ = 1.225 kg m⁻³. See Ulazia et al.
   (2019), *Energy* 187:115938 for the global magnitude of this effect.
3. **Solar+wind mix optimization** — sweep the solar share α of delivered
   energy, report the storage-minimizing mix (α*, S_tot*, f_adj*).
4. **Pooled (interconnected) regions** — forcing-level aggregation to
   continent and world pools quantifies how much interconnection reduces
   storage (including the longitude-spread smoothing of the solar diurnal cycle).
5. **Dunkelflaute statistics** — longest low-resource spell
   (cf < 0.5·mean) in days and the deepest 30-day rolling shortfall.
6. **Storage technology archetypes** — S_tot re-computed for Li-ion
   (0.95/0.95/0.98), pumped hydro (0.85/0.90/0.95) and hydrogen
   (0.60/0.50/1.00) parameter sets (k_R/k_D/k_aS).
7. **Trends** — Mann–Kendall test + Sen slope on annual capacity factors
   (% per decade), exploratory.
8. **Demand scenarios** — flat (headline, the paper's assumption), a 12-value
   monthly profile, or a `degree_day` scenario built from each region's own
   GLDAS air temperature (all-electric heating/cooling around an 18 °C balance
   point) — a globally uniform methodology needing no national load data.

## Running on OSCER (OU)

Storage layout (`hpc/config_oscer.yaml`):

| What | Where | Lifetime |
|---|---|---|
| wget URL list (user-provided) | `/scratch/mchen15/gldas/gldas_list.txt` | input |
| raw 3-hourly granules | `/scratch/mchen15/gldas/raw/<YEAR>/` | deleted after reduction |
| 3-hourly zonal parquets (~1–2 GB total) | `/ourdisk/hpc/caps/mchen15/gldas_full/` | **the durable archive** |
| regions, weights | `/ourdisk/hpc/caps/mchen15/gldas_analysis/data/` | keep |
| results + figures | `/ourdisk/hpc/caps/mchen15/gldas_analysis/{results,figures}/` | copy home when done |

Generate `gldas_list.txt` from the
[GLDAS_NOAH025_3H 2.1 page](https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_3H_2.1/summary)
(Subset/Get Data → Get **Original** Files → download links list) for
2000-01-01 .. 2025-12-31 and drop it in `/scratch/mchen15/gldas/`. Use
*original files*, not the subsetter — the scripts match granule filenames like
`GLDAS_NOAH025_3H.A20000101.0000.021.nc4`.

```bash
# --- one-time setup (login node) ---
# GES DISC uses Earthdata Login token authorization: generate a token at
# urs.earthdata.nasa.gov ("Generate Token", ~60-day lifetime). A current token
# ships in hpc/edl_token.txt -- install it to your home directory:
install -m 600 hpc/edl_token.txt ~/.edl_token
module load Mamba                      # module spider Mamba for the exact name
mamba env create -f hpc/environment.yml
conda activate gldas
mkdir -p /scratch/mchen15/gldas/raw /ourdisk/hpc/caps/mchen15/gldas_full \
         /ourdisk/hpc/caps/mchen15/gldas_analysis/{data,results,figures}

# stage 0: one sample granule, then regions + weights (login node is fine)
grep "\.nc4" /scratch/mchen15/gldas/gldas_list.txt | head -n 1 > /tmp/one.txt
wget --header "Authorization: Bearer $(tr -d '[:space:]' < ~/.edl_token)" \
     -nd -P /scratch/mchen15/gldas/raw -i /tmp/one.txt
python scripts/00_regions.py --config hpc/config_oscer.yaml

# --- pipeline ---
sbatch --array=2000-2025%4 hpc/oscer_01_zonal.sbatch    # wget + reduce, one year/task
# ... wait for the array (check logs/zonal_<year>.out) ...
sbatch hpc/oscer_02_analysis.sbatch                      # analysis + figures
```

Each stage-1 task greps its year's URLs from the list, wgets them into its own
scratch subfolder (`wget -c`, so requeued tasks resume), reduces them to
monthly zonal parquets on `/ourdisk`, and deletes the raw year only if the
download was complete. Reduced months are skipped on rerun.

Notes:

1. The Natural Earth shapefiles are **bundled** in `data/naturalearth/` — copy
   the repo with that folder and nothing on OSCER needs internet except wget.
2. `/scratch` is purged and unbacked — fine, only raw granules live there.
3. Transferring the repo from Windows: run `dos2unix hpc/*.sbatch` first.
   (The token is whitespace-stripped at use, so CRLF in `edl_token.txt` is harmless.)
4. **`hpc/edl_token.txt` is a credential** — it authorizes downloads as your
   Earthdata account. Don't share the repo with that file in it; the token in
   it expires ~2026-08-10, regenerate at urs.earthdata.nasa.gov when needed.
5. The whole downstream analysis runs off the ~1–2 GB zonal parquet archive:
   re-running stages 2–3 with different thresholds, efficiencies, demand
   profiles, or mix grids takes minutes and **never touches the granules**.
   Back that folder up.

## Validate first

Test the pipeline end-to-end with one year and a few countries before the full
submission, e.g. `--array=2025` only, with in `config_oscer.yaml`:

```yaml
regions: { include_countries: [USA, DEU, JPN], ... }   # optional, shrinks everything
```

(If you restrict `include_countries` for the test, rerun stage 0 afterwards
with the full region set — the weight matrix must match before the real run.)

## Configuration highlights

- `regions.admin1_area_km2` (1.5 M km²): countries above this are split into
  admin-1 states/provinces; smaller countries are analyzed whole; tiny island
  states get their nearest land cell.
- `energy.solar.plateau_wm2` (1000 W m⁻² on 3-hourly fluxes ≈ standard test
  insolation) and `energy.wind.start/plateau_ms` (3→12 m s⁻¹): the Eq. 1
  thresholds. **Calibrate** by running CONUS/US states and comparing with the
  paper's Table 1 (solar cf 0.61, wind 0.38 — note those were daily-mean based,
  so 3-hourly values will differ by construction, especially solar).
- `storage`: k_R = 0.9, k_D = 0.8, k_aS = 0.7 exactly as the paper; archetypes
  override per technology.
- `demand.profile`: `flat` | `monthly` | `degree_day`.

## Caveats

1. GLDAS V2.1 is produced with a few months' latency — a too-late `period.end`
   is harmless (empty months are skipped and can be filled in later).
2. The GLDAS grid spans 60°S–90°N; Antarctica is excluded by construction.
3. 26 years samples fewer extreme years than the paper's 40 — the excess
   installation factor f (set by the worst year) reads slightly less
   conservative; worth a sentence when comparing with Table 1.
4. This is a theoretical resource-side analysis: it assumes all incident solar
   or wind over a region's land can be converted. The point is the converse —
   where even this upper bound demands implausible storage or conversion
   fractions, 100% solar/wind self-sufficiency is physically infeasible.
