# auth.py
import os

KEY = os.environ.get('ALPACA_API_KEY')
SECRET = os.environ.get('ALPACA_SECRET_KEY')

if not KEY or not SECRET:
    raise ValueError("API credentials not found in environment variables")