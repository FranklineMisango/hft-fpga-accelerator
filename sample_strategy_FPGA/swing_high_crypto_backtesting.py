import ccxt
import numpy as np
from numba import cuda, njit
import timeit
import csv
from datetime import datetime as dt, timedelta
import pytz
import asyncio
import aiohttp
import cupy as cp
import pandas as pd
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
nest_asyncio.apply()
from openpyxl import Workbook

class data_fetcher():
    def __init__(self):
        self.exchange = ccxt.binance()
        self.initial_gains = {}
        self.data = {}

class SwingHigh():
    def __init__(self):
        self.exchange = ccxt.binance()
        self.initial_gains = {}
        self.data = {}
        self.order_numbers = {}
        self.shares_per_ticker = {}
        self.positions = {}
        self.portfolio_value = 1000  # Initial portfolio value
        self.fees = 0.001  # Trading fee (0.1%)

    async def find_top_gainers(self):
        tickers = self.exchange.fetch_tickers()
        filtered_tickers = [ticker for ticker in tickers.values() if ticker['percentage'] is not None]
        sorted_tickers = sorted(filtered_tickers, key=lambda x: x['percentage'], reverse=True)
        top_gainers = [ticker['symbol'] for ticker in sorted_tickers[:50]]
        return top_gainers

    async def convert_timestamp_ms_to_human_readable(self, timestamp_ms):
        timestamp_s = timestamp_ms / 1000.0
        dt_object = dt.fromtimestamp(timestamp_s)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S')

    async def get_data(self, symbol, since, timeframe='1s', limit=1000):
        all_data = []
        while True:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not data:
                break
            all_data.extend(data)
            since = data[-1][0] + 1  # Move to the next timestamp
            if len(data) < limit:
                break

        # Convert timestamps to human-readable format
        for row in all_data:
            row[0] = await self.convert_timestamp_ms_to_human_readable(row[0])

        return all_data

    async def dynamic_pricing(self, symbol, since, timeframe='1s'):
        data = await self.get_data(symbol, since, timeframe)

        if not data:
            human_readable_since = await self.convert_timestamp_ms_to_human_readable(since)
            print(f"No data fetched for {symbol} since {human_readable_since}")
            return []

        # Fetch the last price for the previous second
        last_price_data = await self.get_data(symbol, int(dt.now().timestamp() * 1000) - 2000, timeframe)
        last_price = last_price_data[-1][4] if last_price_data else data[0][4]

        # Add an additional column named 'last_price' with the last price of the ticker
        for row in data:
            row.append(last_price)
            last_price = row[4]  # Update last price to current row's close price

        return data

    async def fetch_and_save_data(self, symbol, writer):
        user_defined_time_frame = int((dt.now() - timedelta(hours=1)).timestamp() * 1000)
        fetched_data = await self.dynamic_pricing(symbol, user_defined_time_frame, timeframe='1s')

        # Convert the fetched data to a DataFrame
        df = pd.DataFrame(fetched_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'last_price'])

        # Write the DataFrame to a sheet named after the symbol
        df.to_excel(writer, sheet_name=symbol.replace('/', '_'), index=False)

    @staticmethod
    @cuda.jit
    def calculate_gains(initial_prices, current_prices, gains):
        idx = cuda.grid(1)
        if idx < initial_prices.size:
            gains[idx] = (current_prices[idx] - initial_prices[idx]) / initial_prices[idx] * 100

    @staticmethod
    @njit
    def process_data(initial_prices, current_prices):
        gains = np.zeros_like(initial_prices, dtype=np.float32)
        for i in range(initial_prices.size):
            gains[i] = (current_prices[i] - initial_prices[i]) / initial_prices[i] * 100
        return gains

    async def fetch_the_volatile_cryptocurrencies(self, hours):
        start_time = timeit.default_timer()
        top_gainers_list = await self.find_top_gainers()

        # Create a Pandas Excel writer using openpyxl as the engine
        with pd.ExcelWriter('data.xlsx', engine='openpyxl') as writer:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(asyncio.run, self.fetch_and_save_data(symbol, writer)) for symbol in top_gainers_list if '/USDT' in symbol]
                for future in as_completed(futures):
                    future.result()  # Wait for all futures to complete

            # Ensure at least one sheet is visible
            if not writer.book.sheetnames:
                writer.book.create_sheet("Sheet1")

    async def load_volatile_tickers_excel_file(self, file_path):
      try:
          # Load the Excel file
          excel_data = pd.ExcelFile(file_path)
          
          # Initialize lists to store the data
          volatile_tickers = []
          initial_prices = []
          current_prices = []
          symbols = []
          
          since = int((dt.now() - timedelta(hours=1)).timestamp() * 1000) # CHANGE HERE AS WELL TO 1 HR

          # Iterate through each sheet in the Excel file
          for sheet_name in excel_data.sheet_names:
              # Load the sheet into a DataFrame
              df = pd.read_excel(file_path, sheet_name=sheet_name)
              
              # Extract the necessary information
              latest_price = df['last_price'].iloc[-1]
              close_price = df['close'].iloc[-1]
              
              # Append the data to the respective lists
              volatile_tickers.append({
                  'symbol': sheet_name,
                  'initial_price': latest_price,
                  'current_price': close_price,
                  '%change': (close_price - latest_price) / latest_price * 100,
                  'num_trades': 0  # Initialize num_trades with 0
              })
              initial_prices.append(latest_price)
              current_prices.append(close_price)
              symbols.append(sheet_name.split('_')[0])  # Assuming the symbol is the part before '_USDT'

          # Convert to CuPy arrays for GPU processing
          initial_prices = cp.array(initial_prices, dtype=cp.float32)
          current_prices = cp.array(current_prices, dtype=cp.float32)
          gains = cp.zeros_like(initial_prices)

          threads_per_block = 128
          blocks_per_grid = (initial_prices.size + (threads_per_block - 1)) // threads_per_block
          self.calculate_gains[blocks_per_grid, threads_per_block](initial_prices, current_prices, gains)

          # Copy gains back to CPU from GPU
          gains = cp.asnumpy(gains)

          for i, symbol in enumerate(symbols):
              gain = gains[i]
              edited_symbol = symbol + '/USDT'
              num_trades = self.exchange.fetch_trades(edited_symbol, since=since)
              volatile_tickers[i]['num_trades'] = len(num_trades)  # Update num_trades with the actual number of trades
              if gain >= 2:
                  self.initial_gains[symbol] = gain
              elif symbol in self.initial_gains and gain < self.initial_gains[symbol] * 0.95:
                  volatile_tickers = [ticker for ticker in volatile_tickers if ticker['symbol'] != symbol]
                  del self.initial_gains[symbol]

          volatile_tickers.sort(key=lambda x: x['%change'], reverse=True)
          with open('volatile_tickers.csv', 'w') as f:
              writer = csv.writer(f)
              writer.writerow(['symbol', 'initial_price', 'current_price', '%change', 'num_trades'])
              for ticker in volatile_tickers:
                  writer.writerow([ticker['symbol'], ticker['initial_price'], ticker['current_price'], ticker['%change'], ticker['num_trades']])
          return volatile_tickers

      except Exception as e:
          print(f"An error occurred while loading the Excel file: {e}")
          return None

    async def log_message(self, message):
        print(message)
        with open('backtest_log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([dt.now(), message])

    async def get_position(self, symbol):
        return self.positions.get(symbol, False)

    async def get_last_price(self, symbol):
        try:
            excel_data = pd.ExcelFile('data.xlsx')
            df = pd.read_excel('data.xlsx', sheet_name=symbol.replace('/', '_'))
            current_price = df['last_price'].iloc[-1]
            return current_price
        except Exception as e:
            print(f"An error occurred while fetching the last price for {symbol}: {e}")
            return None

    async def sell_all(self, symbol, entry_price):
        current_price = await self.get_last_price(symbol)
        if current_price is not None and await self.get_position(symbol):
            dropping_price = entry_price * 0.995
            higher_than_earlier_price = entry_price * 1.015
            if current_price < dropping_price or current_price >= higher_than_earlier_price:
                shares = self.shares_per_ticker[symbol]
                sale_value = shares * current_price
                sale_value -= sale_value * self.fees  # Subtract fees
                self.portfolio_value += sale_value
                await self.log_message(f"Selling all for {symbol} at {current_price}")
                self.positions[symbol] = False

    async def run_backtest(self):
        fetch_Caller = await self.fetch_the_volatile_cryptocurrencies(hours=1) # Fetch for 1 HOUR
        volatile_tickers = await self.load_volatile_tickers_excel_file(file_path="data.xlsx")
        if volatile_tickers is None:
            print("No volatile tickers found.")
            return

        self.symbols = [ticker['symbol'] for ticker in volatile_tickers]

        # Allocate 30% to the highest volatility ticker and 70% to the rest
        if volatile_tickers:
            highest_volatility_ticker = volatile_tickers[0]
            highest_volatility_allocation = self.portfolio_value * 0.3
            rest_allocation = self.portfolio_value * 0.7 / (len(volatile_tickers) - 1) if len(volatile_tickers) > 1 else 0

        for ticker in volatile_tickers:
            symbol = ticker['symbol']
            initial_price_trading = ticker['initial_price']
            allocation = highest_volatility_allocation if symbol == highest_volatility_ticker['symbol'] else rest_allocation
            shares = allocation / initial_price_trading
            self.shares_per_ticker[symbol] = shares
            self.positions[symbol] = True
            self.data[symbol] = []  # Initialize the data list for the symbol
            await self.log_message(f"Bought {shares} coins of {symbol} at {initial_price_trading}")

        for _ in range(60):
            for symbol in self.symbols:
                if await self.get_position(symbol):
                    current_price = await self.get_last_price(symbol)
                    entry_price = self.data[symbol][0] if symbol in self.data and self.data[symbol] else current_price
                    self.data[symbol].append(current_price)
                    if current_price < entry_price * 0.995 or current_price >= entry_price * 1.015:
                        await self.sell_all(symbol, entry_price)

        # Sell everything at the end of the backtest
        for symbol in self.symbols:
            if await self.get_position(symbol):
                await self.sell_all(symbol, self.data[symbol][0])

        # Calculate final portfolio value
        final_portfolio_value = 0
        for symbol in self.symbols:
            if symbol in self.shares_per_ticker:
                final_portfolio_value += self.shares_per_ticker[symbol] * await self.get_last_price(symbol)
        final_portfolio_value -= final_portfolio_value * self.fees  # Subtract fees

        await self.log_message(f"Final portfolio value: {final_portfolio_value}")

if __name__ == "__main__":
    start_time = timeit.default_timer()
    strategy = SwingHigh()
    asyncio.run(strategy.run_backtest())
    elapsed = timeit.default_timer() - start_time
    print(f"Backtest completed in {elapsed:.2f} seconds.")