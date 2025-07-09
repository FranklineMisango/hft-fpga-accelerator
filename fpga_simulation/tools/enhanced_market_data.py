#!/usr/bin/env python3
"""
Enhanced Market Data Feed Simulator
Support for realistic trading protocols including ITCH, FIX, and OUCH.
"""

import sys
import os
import time
import struct
import socket
import threading
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ProtocolType(Enum):
    """Supported market data protocols"""
    ITCH_5_0 = "ITCH_5.0"
    FIX_4_4 = "FIX_4.4"
    OUCH_4_2 = "OUCH_4.2"
    CUSTOM = "CUSTOM"

@dataclass
class MarketDataMessage:
    """Market data message structure"""
    timestamp: datetime
    symbol: str
    message_type: str
    protocol: ProtocolType
    raw_data: bytes
    parsed_data: Dict[str, Any]
    sequence_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'message_type': self.message_type,
            'protocol': self.protocol.value,
            'parsed_data': self.parsed_data,
            'sequence_number': self.sequence_number,
            'raw_data_length': len(self.raw_data)
        }

class ITCHProtocolSimulator:
    """NASDAQ ITCH 5.0 protocol simulator"""
    
    def __init__(self):
        self.sequence_number = 0
        self.reference_numbers = {}
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # ITCH message types
        self.message_types = {
            'S': 'System Event',
            'R': 'Stock Directory',
            'H': 'Stock Trading Action',
            'Y': 'Reg SHO Restriction',
            'L': 'Market Participant Position',
            'V': 'MWCB Decline Level',
            'W': 'MWCB Breach',
            'K': 'IPO Quoting Period Update',
            'J': 'LULD Auction Collar',
            'A': 'Add Order',
            'F': 'Add Order - MPID Attribution',
            'E': 'Order Executed',
            'C': 'Order Executed With Price',
            'X': 'Order Cancel',
            'D': 'Order Delete',
            'U': 'Order Replace',
            'P': 'Trade',
            'Q': 'Cross Trade',
            'B': 'Broken Trade',
            'I': 'NOII'
        }
    
    def generate_message(self, symbol: str = None) -> MarketDataMessage:
        """Generate a random ITCH message"""
        if not symbol:
            symbol = random.choice(self.symbols)
        
        # Common message types for trading
        common_types = ['A', 'E', 'P', 'X', 'D']
        message_type = random.choice(common_types)
        
        timestamp = datetime.now()
        self.sequence_number += 1
        
        # Generate message based on type
        if message_type == 'A':  # Add Order
            return self._generate_add_order(symbol, timestamp)
        elif message_type == 'E':  # Order Executed
            return self._generate_order_executed(symbol, timestamp)
        elif message_type == 'P':  # Trade
            return self._generate_trade(symbol, timestamp)
        elif message_type == 'X':  # Order Cancel
            return self._generate_order_cancel(symbol, timestamp)
        elif message_type == 'D':  # Order Delete
            return self._generate_order_delete(symbol, timestamp)
        else:
            return self._generate_add_order(symbol, timestamp)
    
    def _generate_add_order(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Add Order message (Type A)"""
        reference_number = random.randint(1000000, 9999999)
        self.reference_numbers[reference_number] = symbol
        
        # ITCH Add Order format
        price = int(random.uniform(100, 200) * 10000)  # Price in 1/10000 increments
        shares = random.randint(100, 10000)
        buy_sell = random.choice(['B', 'S'])
        
        # Pack binary data (simplified ITCH format)
        raw_data = struct.pack('>cHQ8sIcI',
                              b'A',  # Message type
                              39,     # Length
                              reference_number,  # Stock locate
                              symbol.ljust(8).encode()[:8],  # Stock symbol
                              shares,  # Shares
                              buy_sell.encode(),  # Buy/Sell indicator
                              price   # Price
                              )
        
        parsed_data = {
            'reference_number': reference_number,
            'symbol': symbol,
            'shares': shares,
            'buy_sell': buy_sell,
            'price': price / 10000.0,
            'side': 'BUY' if buy_sell == 'B' else 'SELL'
        }
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Add Order',
            protocol=ProtocolType.ITCH_5_0,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )
    
    def _generate_order_executed(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Order Executed message (Type E)"""
        if not self.reference_numbers:
            # Generate an add order first
            self._generate_add_order(symbol, timestamp)
        
        reference_number = random.choice(list(self.reference_numbers.keys()))
        executed_shares = random.randint(100, 1000)
        match_number = random.randint(1000000, 9999999)
        
        raw_data = struct.pack('>cHQIQ',
                              b'E',  # Message type
                              31,     # Length
                              reference_number,  # Order reference
                              executed_shares,   # Executed shares
                              match_number       # Match number
                              )
        
        parsed_data = {
            'reference_number': reference_number,
            'executed_shares': executed_shares,
            'match_number': match_number,
            'symbol': symbol
        }
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Order Executed',
            protocol=ProtocolType.ITCH_5_0,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )
    
    def _generate_trade(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Trade message (Type P)"""
        reference_number = random.randint(1000000, 9999999)
        price = int(random.uniform(100, 200) * 10000)
        shares = random.randint(100, 10000)
        match_number = random.randint(1000000, 9999999)
        
        raw_data = struct.pack('>cHQ8sIIQ',
                              b'P',  # Message type
                              44,     # Length
                              reference_number,  # Order reference
                              symbol.ljust(8).encode()[:8],  # Stock symbol
                              shares,  # Shares
                              price,   # Price
                              match_number  # Match number
                              )
        
        parsed_data = {
            'reference_number': reference_number,
            'symbol': symbol,
            'shares': shares,
            'price': price / 10000.0,
            'match_number': match_number
        }
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Trade',
            protocol=ProtocolType.ITCH_5_0,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )
    
    def _generate_order_cancel(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Order Cancel message (Type X)"""
        if not self.reference_numbers:
            self._generate_add_order(symbol, timestamp)
        
        reference_number = random.choice(list(self.reference_numbers.keys()))
        cancelled_shares = random.randint(100, 1000)
        
        raw_data = struct.pack('>cHQI',
                              b'X',  # Message type
                              23,     # Length
                              reference_number,  # Order reference
                              cancelled_shares   # Cancelled shares
                              )
        
        parsed_data = {
            'reference_number': reference_number,
            'cancelled_shares': cancelled_shares,
            'symbol': symbol
        }
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Order Cancel',
            protocol=ProtocolType.ITCH_5_0,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )
    
    def _generate_order_delete(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Order Delete message (Type D)"""
        if not self.reference_numbers:
            self._generate_add_order(symbol, timestamp)
        
        reference_number = random.choice(list(self.reference_numbers.keys()))
        
        raw_data = struct.pack('>cHQ',
                              b'D',  # Message type
                              19,     # Length
                              reference_number   # Order reference
                              )
        
        parsed_data = {
            'reference_number': reference_number,
            'symbol': symbol
        }
        
        # Remove from reference numbers
        if reference_number in self.reference_numbers:
            del self.reference_numbers[reference_number]
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Order Delete',
            protocol=ProtocolType.ITCH_5_0,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )

class FIXProtocolSimulator:
    """FIX 4.4 protocol simulator"""
    
    def __init__(self):
        self.sequence_number = 0
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    def generate_message(self, symbol: str = None) -> MarketDataMessage:
        """Generate a random FIX message"""
        if not symbol:
            symbol = random.choice(self.symbols)
        
        timestamp = datetime.now()
        self.sequence_number += 1
        
        # Generate market data snapshot
        return self._generate_market_data_snapshot(symbol, timestamp)
    
    def _generate_market_data_snapshot(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Market Data Snapshot (MsgType W)"""
        bid_price = random.uniform(100, 200)
        ask_price = bid_price + random.uniform(0.01, 0.10)
        bid_size = random.randint(100, 10000)
        ask_size = random.randint(100, 10000)
        
        # FIX message format
        fix_message = (
            f"8=FIX.4.4|9=150|35=W|49=SENDER|56=TARGET|34={self.sequence_number}|"
            f"52={timestamp.strftime('%Y%m%d-%H:%M:%S')}|55={symbol}|"
            f"268=2|269=0|270={bid_price:.4f}|271={bid_size}|"
            f"269=1|270={ask_price:.4f}|271={ask_size}|10=123|"
        )
        
        raw_data = fix_message.encode()
        
        parsed_data = {
            'symbol': symbol,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'spread': ask_price - bid_price,
            'mid_price': (bid_price + ask_price) / 2
        }
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Market Data Snapshot',
            protocol=ProtocolType.FIX_4_4,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )

class OUCHProtocolSimulator:
    """OUCH 4.2 protocol simulator"""
    
    def __init__(self):
        self.sequence_number = 0
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    def generate_message(self, symbol: str = None) -> MarketDataMessage:
        """Generate a random OUCH message"""
        if not symbol:
            symbol = random.choice(self.symbols)
        
        timestamp = datetime.now()
        self.sequence_number += 1
        
        # Generate order acknowledgment
        return self._generate_order_accepted(symbol, timestamp)
    
    def _generate_order_accepted(self, symbol: str, timestamp: datetime) -> MarketDataMessage:
        """Generate Order Accepted message (Type A)"""
        order_token = f"ORDER{self.sequence_number:06d}"
        order_state = 'L'  # Live
        
        # OUCH binary format
        raw_data = struct.pack('>c14sc8sIIc',
                              b'A',  # Message type
                              order_token.encode()[:14],  # Order token
                              order_state.encode(),  # Order state
                              symbol.ljust(8).encode()[:8],  # Stock symbol
                              random.randint(100, 10000),  # Shares
                              int(random.uniform(100, 200) * 10000),  # Price
                              random.choice([b'B', b'S'])  # Buy/Sell
                              )
        
        parsed_data = {
            'order_token': order_token,
            'order_state': order_state,
            'symbol': symbol,
            'shares': struct.unpack('>I', raw_data[23:27])[0],
            'price': struct.unpack('>I', raw_data[27:31])[0] / 10000.0,
            'side': raw_data[31:32].decode()
        }
        
        return MarketDataMessage(
            timestamp=timestamp,
            symbol=symbol,
            message_type='Order Accepted',
            protocol=ProtocolType.OUCH_4_2,
            raw_data=raw_data,
            parsed_data=parsed_data,
            sequence_number=self.sequence_number
        )

class EnhancedMarketDataFeed:
    """Enhanced market data feed with multiple protocol support"""
    
    def __init__(self, protocol: ProtocolType = ProtocolType.ITCH_5_0):
        self.protocol = protocol
        self.running = False
        self.feed_thread = None
        self.message_callback = None
        self.latency_simulation = True
        self.burst_mode = False
        
        # Initialize protocol simulators
        self.simulators = {
            ProtocolType.ITCH_5_0: ITCHProtocolSimulator(),
            ProtocolType.FIX_4_4: FIXProtocolSimulator(),
            ProtocolType.OUCH_4_2: OUCHProtocolSimulator()
        }
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'bytes_sent': 0,
            'start_time': None,
            'last_message_time': None,
            'message_rate': 0,
            'protocol_breakdown': {}
        }
        
        # Configuration
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        self.message_rate_hz = 1000  # Messages per second
        self.latency_range_ns = (50, 200)  # Simulated network latency
    
    def set_message_callback(self, callback):
        """Set callback function for message processing"""
        self.message_callback = callback
    
    def start_feed(self, symbols: List[str] = None):
        """Start the market data feed"""
        if self.running:
            print("⚠️  Market data feed already running")
            return
        
        if symbols:
            self.symbols = symbols
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        self.stats['messages_sent'] = 0
        self.stats['bytes_sent'] = 0
        
        self.feed_thread = threading.Thread(target=self._feed_loop)
        self.feed_thread.daemon = True
        self.feed_thread.start()
        
        print(f"🚀 Market data feed started")
        print(f"   Protocol: {self.protocol.value}")
        print(f"   Symbols: {len(self.symbols)}")
        print(f"   Message rate: {self.message_rate_hz} Hz")
    
    def stop_feed(self):
        """Stop the market data feed"""
        if not self.running:
            print("⚠️  Market data feed not running")
            return
        
        self.running = False
        if self.feed_thread:
            self.feed_thread.join()
        
        duration = datetime.now() - self.stats['start_time']
        print(f"⏹️  Market data feed stopped after {duration.total_seconds():.1f} seconds")
        self._print_statistics()
    
    def _feed_loop(self):
        """Main feed loop"""
        simulator = self.simulators[self.protocol]
        interval = 1.0 / self.message_rate_hz
        
        while self.running:
            try:
                # Generate message
                symbol = random.choice(self.symbols)
                message = simulator.generate_message(symbol)
                
                # Simulate network latency
                if self.latency_simulation:
                    latency_ns = random.uniform(*self.latency_range_ns)
                    time.sleep(latency_ns / 1e9)
                
                # Process message
                if self.message_callback:
                    self.message_callback(message)
                
                # Update statistics
                self._update_statistics(message)
                
                # Control message rate
                if not self.burst_mode:
                    time.sleep(interval)
                
            except Exception as e:
                print(f"❌ Feed error: {e}")
                break
    
    def _update_statistics(self, message: MarketDataMessage):
        """Update feed statistics"""
        self.stats['messages_sent'] += 1
        self.stats['bytes_sent'] += len(message.raw_data)
        self.stats['last_message_time'] = message.timestamp
        
        # Calculate message rate
        if self.stats['start_time']:
            duration = (message.timestamp - self.stats['start_time']).total_seconds()
            if duration > 0:
                self.stats['message_rate'] = self.stats['messages_sent'] / duration
        
        # Protocol breakdown
        protocol_str = message.protocol.value
        if protocol_str not in self.stats['protocol_breakdown']:
            self.stats['protocol_breakdown'][protocol_str] = 0
        self.stats['protocol_breakdown'][protocol_str] += 1
    
    def _print_statistics(self):
        """Print feed statistics"""
        print("\n" + "="*50)
        print("📊 MARKET DATA FEED STATISTICS")
        print("="*50)
        print(f"Messages sent: {self.stats['messages_sent']:,}")
        print(f"Bytes sent: {self.stats['bytes_sent']:,}")
        print(f"Message rate: {self.stats['message_rate']:.1f} Hz")
        print(f"Protocol: {self.protocol.value}")
        print("="*50)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()
    
    def set_burst_mode(self, enabled: bool):
        """Enable/disable burst mode"""
        self.burst_mode = enabled
        if enabled:
            print("🔥 Burst mode enabled")
        else:
            print("⏸️  Burst mode disabled")
    
    def set_latency_simulation(self, enabled: bool, latency_range_ns: Tuple[int, int] = None):
        """Enable/disable latency simulation"""
        self.latency_simulation = enabled
        if latency_range_ns:
            self.latency_range_ns = latency_range_ns
        
        if enabled:
            print(f"⚡ Latency simulation enabled: {self.latency_range_ns[0]}-{self.latency_range_ns[1]} ns")
        else:
            print("⚡ Latency simulation disabled")

def main():
    """Main function for testing the enhanced market data feed"""
    print("🚀 Enhanced Market Data Feed Test")
    print("=================================")
    
    # Test callback function
    def message_handler(message: MarketDataMessage):
        print(f"📨 {message.protocol.value} | {message.message_type} | {message.symbol} | "
              f"Seq: {message.sequence_number} | Data: {message.parsed_data}")
    
    # Test different protocols
    protocols = [ProtocolType.ITCH_5_0, ProtocolType.FIX_4_4, ProtocolType.OUCH_4_2]
    
    for protocol in protocols:
        print(f"\n🔬 Testing {protocol.value}:")
        
        # Create feed
        feed = EnhancedMarketDataFeed(protocol=protocol)
        feed.set_message_callback(message_handler)
        
        # Configure feed
        feed.set_latency_simulation(True, (100, 500))
        
        # Start feed
        feed.start_feed(['AAPL', 'GOOGL', 'MSFT'])
        
        # Run for a few seconds
        time.sleep(3)
        
        # Stop feed
        feed.stop_feed()
    
    print("\n✅ All protocol tests completed!")

if __name__ == "__main__":
    main()
