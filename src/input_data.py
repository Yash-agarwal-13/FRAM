"""Locate the five assignment input workbooks locally or in Google Colab."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_INPUTS = ROOT / 'data/raw/input_files'

MARKET_FILE = 'Biocon_data_updated.xlsx'
FINANCIAL_FILE = 'Financial_Statements (2) (1).xlsx'
OPTIONS_FILE = 'Biocon_options-_data.xlsx'
RATE_91_FILE = 'Auctions of 91-Day Government of India Treasury Bills.xlsx'
RATE_364_FILE = 'Auctions of 364-Day Government of India Treasury Bills.xlsx'


def file(name):
    """Return an input path from the project, Colab, cwd or an override folder."""
    folders = [
        Path(os.environ['FRAM_INPUT_DIR']) if os.getenv('FRAM_INPUT_DIR') else None,
        LOCAL_INPUTS,
        Path.cwd(),
        Path('/content'),
        Path.home() / 'Desktop',
    ]
    for folder in filter(None, folders):
        candidate = folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Missing input file: {name}')


def all_files():
    names = [MARKET_FILE, FINANCIAL_FILE, OPTIONS_FILE,
             RATE_91_FILE, RATE_364_FILE]
    return {name: file(name) for name in names}
