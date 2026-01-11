# Historical SMI Member Data

This directory contains CSV files for former SMI (Swiss Market Index) constituents.

## Purpose

These files are used to correct **survivorship bias** in backtests by including companies that were removed from the index during the analysis period (2010-2024).

## Expected Files

Place the following CSV files in this directory:

| Filename    | Ticker   | Company               | Exit Date   | Reason                    |
|-------------|----------|----------------------|-------------|---------------------------|
| `CSGN.csv`  | CSGN.SW  | Credit Suisse        | 2023-06-13  | Acquired by UBS           |
| `SGSN.csv`  | SGSN.SW  | SGS                  | 2022-09-19  | Removed from SMI          |
| `UHR.csv`   | UHR.SW   | Swatch Group         | 2021-09-20  | Removed from SMI          |
| `ADEN.csv`  | ADEN.SW  | Adecco               | 2020-09-21  | Removed from SMI          |
| `BAER.csv`  | BAER.SW  | Julius Bär           | 2019-04-10  | Removed from SMI          |
| `SYNN.csv`  | SYNN.SW  | Syngenta             | 2017-05-15  | Acquired by ChemChina     |
| `ATLN.csv`  | ATLN.SW  | Actelion             | 2017-05-03  | Acquired by J&J           |
| `RIGN.csv`  | RIGN.SW  | Transocean           | 2016-03-21  | Removed from SMI          |
| `SYST.csv`  | SYST.SW  | Synthes              | 2012-06-18  | Acquired by J&J           |
| `LONN.csv`  | LONN.SW  | Lonza (historical)   | 2011-09-19  | Exited, later re-entered  |
| `SLHN.csv`  | SLHN.SW  | Swiss Life (hist.)   | 2010-06-21  | Exited, later re-entered  |

## CSV Format

Files must follow Yahoo Finance format:

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2010-01-04,52.50,53.20,52.10,52.85,48.32,1234567
2010-01-05,52.90,54.00,52.80,53.75,49.14,2345678
...
```

### Requirements

- **Index column**: `Date` (YYYY-MM-DD format)
- **Required columns**: `Close` (or `Adj Close`)
- **Optional columns**: `Open`, `High`, `Low`, `Volume`
- **Encoding**: UTF-8
- **Date range**: Should cover the period the company was in the SMI

### Column Handling

The loader automatically:
1. Renames `Adj Close` to `Close` if present
2. Parses dates flexibly
3. Sorts by date ascending

## Data Truncation (Critical!)

**Data is automatically truncated at the exit date** to prevent look-ahead bias.

For example, if you provide Credit Suisse data up to 2024, only data up to 2023-06-13 (exit date) will be used in analysis.

This ensures the backtest doesn't "know" about future events (delistings, acquisitions).

## Data Sources

Historical data can be obtained from:
- Yahoo Finance (historical downloads)
- Bloomberg Terminal
- Refinitiv Eikon
- SIX Swiss Exchange archives

## Validation

Run the following to check CSV file status:

```python
from src.data_loader import DataLoader

loader = DataLoader(include_historical=True)
print(loader.get_historical_members_info())
```
