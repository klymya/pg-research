# Footfall report

Turns a weekly CSV of hourly IN/OUT counts per floor into a validated daily report:
KPIs, hourly detail, daily totals per floor, daily totals overall, and an explicit
data-quality section. Nothing invalid is silently dropped or "fixed".

## Install

Python 3.9+ is required (get it from <https://python.org> — tick **Add python.exe to PATH**).

### Windows (project in `C:\Users\maorn\footfall_report`)

In PowerShell:

```powershell
cd C:\Users\maorn\footfall_report
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, either run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use `cmd.exe` and
`.\.venv\Scripts\activate.bat`.

### macOS / Linux

```bash
cd footfall_report
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the project folder, with the virtual environment active:

```powershell
streamlit run app.py
```

The app opens at <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop it.

On Windows you can also double-click **`run_windows.bat`**, which creates the virtual
environment on first use, installs the dependencies and starts the app.

Run the tests (optional):

```powershell
pip install pytest
pytest -q
```

## Use

1. **Upload** the CSV (`1. Upload the CSV file`).
2. **Check the column mapping** (step 2). Columns are detected automatically from the
   names listed in `config.yaml`; if the detection is wrong, correct it in the
   drop-downs. The delimiter and the detected time format are shown above them.
3. **Press `Process file`** (step 3). A green banner confirms the file was processed and
   reports how many rows were read, accepted and rejected.
4. Read the report:
   * **Summary** - Total IN, Total OUT, IN−OUT, error rate, floors, days, busiest day, busiest hour.
   * **Daily totals** - one row per date, across all floors.
   * **Daily totals by floor** - one row per date and floor.
   * **Hourly detail** - Date | Hour | Floor | IN | OUT, exactly as read from the file.
   * **Data quality** - every finding, plus the rejected rows shown unchanged with the
     reason each one failed.

Every table has a `Download as CSV` button.

Three example files are in `sample_data/` (a clean one, one full of deliberate problems,
and one with different column names, a `;` delimiter and a single timestamp column).

## Error rate

The requested definition, "in events + out events / the total of in+out events", is
literally 1, so it was read as the intended meaning: **the IN/OUT imbalance**.

```
error rate = |Total IN − Total OUT| / (Total IN + Total OUT) × 100
```

Everyone who enters eventually leaves, so over a full day IN and OUT should converge.
Whatever is left over is unmatched — i.e. miscounted. It is shown as a KPI (whole file),
per date, and per date and floor. When a floor/day recorded no events at all the rate is
undefined and shown as `-`, never `0%`.

If you meant a different formula, it lives in one function: `error_rate()` in
`footfall/analysis.py`.

## Configuration — `config.yaml`

No dates, floors, column names or thresholds are hard-coded. Edit `config.yaml`; the same
options can be overridden per session in the app sidebar.

| Key | Purpose |
| --- | --- |
| `time_mode` | `split` (date + hour columns), `combined` (one timestamp column), `auto` |
| `columns` | Accepted header names per field. Matching ignores case, spaces and punctuation, so `Floor Name` matches `floor_name`. Put the header your export uses first. |
| `date_formats` | Explicit `strptime` formats tried before automatic parsing, e.g. `["%d/%m/%Y"]` |
| `dayfirst` | Is `03/04/2026` the 3rd of April (`true`) or the 4th of March (`false`)? |
| `allowed_floors` | Expected floor labels. Empty = accept anything; filled in = any other label is rejected and reported |
| `allow_sub_hourly` | `false`: a `10:30` timestamp is rejected. `true`: it is bucketed into the 10:00 hour and reported |
| `missing_tokens` | Cell values treated as missing (`n/a`, `null`, `-`, …) |
| `counts.allow_negative` | Whether a negative count is valid |
| `counts.require_integer` | Whether `5.5` is valid |
| `counts.max_plausible_hourly_count` | Optional upper sanity bound per hour per floor |
| `csv.delimiter` / `csv.encoding` | Leave the delimiter empty to auto-detect |

To add a new spelling of a column, add it to the relevant list under `columns:` —
no code change.

## Assumptions about the CSV

* One header row; one row per date × hour × floor.
* Time is stored either as a date column plus an hour column, or as a single timestamp
  column. Both are detected automatically.
* Hours are accepted as `8`, `08`, `08:00`, `08:00:00`, `8am`, `13h` or `08:00-09:00`
  (the start of the range is used).
* Dates are parsed with the formats in `date_formats` first, otherwise automatically;
  `dayfirst` decides ambiguous `03/04/2026` cases.
* Counts are whole, non-negative numbers; `1,234` is accepted as 1234. A decimal comma
  (`12,5` meaning 12.5) is *not* supported.
* IN and OUT are counts for that hour, not running totals.
* All timestamps are in one timezone; no DST handling is attempted.
* Columns not used by the report are ignored and listed as an info finding.

## What is validated

Rejected rows (excluded from every total, listed with a per-row reason):

* missing values in any required field
* unparseable dates, timestamps or hours; hour outside 0–23
* timestamps not on the hour (unless `allow_sub_hourly` is on)
* non-numeric, fractional, negative or implausibly large counts
* a floor label that is not in `allowed_floors` (when that list is configured)
* conflicting duplicates: the same date/hour/floor reported twice with different counts —
  the correct value cannot be guessed, so all copies are excluded

Reported without dropping data:

* exact duplicate rows (one copy is kept for the totals, the rest are listed)
* duplicate or unnamed columns in the header
* floor labels that differ only in case or punctuation (`Floor 1` vs `floor 1`)
* expected floors with no data in the file
* floor/day combinations missing hours that other floors report the same day
* calendar days with no data between the first and last date
* floor/days that recorded zero IN and zero OUT all day (possible sensor outage)
* floor/days with an error rate above 25%
* columns not used by the report

Errors that stop the report: a required column that cannot be mapped, an empty file, a
file with a header but no rows, or no rows passing validation.

## Structure

```
footfall_report/
├── app.py                  # Streamlit UI: upload -> mapping -> process -> report
├── config.yaml             # CSV layout + validation rules (the only thing you edit)
├── requirements.txt
├── footfall/
│   ├── config.py           # loads config.yaml into a dataclass
│   ├── parsing.py          # read, map columns, validate every row, de-duplicate
│   ├── analysis.py         # aggregations, KPIs, dataset-level quality checks
│   └── issues.py           # data-quality finding type
├── sample_data/            # three example CSVs
└── tests/test_pipeline.py  # pytest suite
```

`footfall/` has no Streamlit dependency, so the same pipeline can be reused from a script,
a scheduled job or a different UI.
