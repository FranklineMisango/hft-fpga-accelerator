import ccxt
from prettytable import PrettyTable
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from numba import cuda, float32
import numpy as np
import threading
import nest_asyncio
import websockets
import json

nest_asyncio.apply()

class FindTopGainers():
    def __init__(self):
        self.exchange = ccxt.binance()
        self.data = {}
        self.top_gainers = []
        self.lock = threading.Lock()

    async def fetch_recent_trades(self, symbol):
        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
        trades = []

        async with websockets.connect(uri) as websocket:
            end_time = time.time() + 60  # Collect trades for 1 minute
            while time.time() < end_time:
                response = await websocket.recv()
                trade = json.loads(response)
                trades.append(trade)
        
        return trades

    @staticmethod
    @cuda.jit
    def calculate_percentage_change_kernel(start_price, end_price, result):
        idx = cuda.grid(1)
        if idx < start_price.size:
            result[idx] = ((end_price[idx] - start_price[idx]) / start_price[idx]) * 100

    def calculate_percentage_change(self, trades):
        if not trades:
            return 0
        start_price = np.array([trades[0]['p']], dtype=np.float32)
        end_price = np.array([trades[-1]['p']], dtype=np.float32)
        result = np.zeros_like(start_price)

        threads_per_block = 128
        blocks_per_grid = (start_price.size + (threads_per_block - 1)) // threads_per_block

        FindTopGainers.calculate_percentage_change_kernel[blocks_per_grid, threads_per_block](start_price, end_price, result)

        percentage_change = result[0]  # Get the calculated percentage change
        return percentage_change

    async def find_top_gainers(self):
        tickers = self.exchange.fetch_tickers()
        symbols = [ticker['symbol'] for ticker in tickers.values() if ticker['percentage'] is not None and '/USDT' in ticker['symbol']]
        percentage_changes = []

        tasks = [self.fetch_recent_trades(symbol) for symbol in symbols]
        trades_list = await asyncio.gather(*tasks)

        for symbol, trades in zip(symbols, trades_list):
            percentage_change = self.calculate_percentage_change(trades)
            percentage_changes.append((symbol, tickers[symbol]['last'], percentage_change))

        sorted_tickers = sorted(percentage_changes, key=lambda x: x[2], reverse=True)
        with self.lock:
            self.top_gainers = sorted_tickers[:5]

        table = PrettyTable()
        table.field_names = ["Symbol", "Price", "Percentage Change (5 min)"]
        for ticker in self.top_gainers:
            table.add_row([ticker[0], ticker[1], ticker[2]])
        print(table)

    async def update_ticker_list(self):
        while True:
            await self.find_top_gainers()
            await asyncio.sleep(60)  # Wait for 1 minute before fetching again

    def run(self, duration_minutes):
        print("Starting to find top gainers...")
        loop = asyncio.get_event_loop()
        end_time = time.time() + duration_minutes * 60

        async def stop_loop_after_duration():
            while time.time() < end_time:
                await asyncio.sleep(1)
            for task in asyncio.all_tasks(loop):
                task.cancel()

        loop.run_until_complete(asyncio.gather(self.update_ticker_list(), stop_loop_after_duration()))

if __name__ == "__main__":
    strategy = FindTopGainers()
    strategy.run(duration_minutes=1)  # Run the algorithm for 1 minute