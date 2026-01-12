try:
    from auth_local import KEY, SECRET
except ImportError:
    import os
    KEY = os.getenv('ALPACA_API_KEY')
    SECRET = os.getenv('ALPACA_SECRET_KEY')
    if not KEY or not SECRET:
        raise ValueError("API credentials not found in environment variables")