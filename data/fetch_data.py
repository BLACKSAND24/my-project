import random

def fetch_market_data(window_size=120):
    returns = [random.gauss(0.0001, 0.01) for _ in range(window_size)]
    return {"returns": returns}
