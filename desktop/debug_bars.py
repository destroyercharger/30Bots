import sys
sys.path.insert(0, '.')
from data.broker_adapter import AlpacaBroker
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
import pandas as pd

broker = AlpacaBroker(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
bars = broker.get_bars('AAPL', '5Min', limit=100)
print(f'Got {len(bars)} bars')
if bars:
    print(f'First bar keys: {bars[0].keys()}')
    df = pd.DataFrame(bars)
    print(f'DataFrame shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(df.tail(3))
