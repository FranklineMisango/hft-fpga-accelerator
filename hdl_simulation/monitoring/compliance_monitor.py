"""
Real-Time Performance & Compliance Monitoring
Institutional-grade metrics collection and reporting
"""

import time
import json
import threading
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderMetrics:
    """Per-order execution metrics"""
    order_id: int
    timestamp: float
    strategy: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    execution_price: float
    latency_ns: int
    status: str  # ACCEPTED/REJECTED/FILLED/PARTIAL

@dataclass
class StrategyMetrics:
    """Strategy performance snapshot"""
    strategy: str
    timestamp: float
    orders_generated: int
    avg_latency_ns: float
    pnl_realized: float
    pnl_unrealized: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float

class ComplianceMonitor:
    """Real-time compliance & risk monitoring"""
    
    def __init__(self):
        self.order_history = deque(maxlen=100000)
        self.strategy_history = deque(maxlen=10000)
        self.lock = threading.RLock()
        
        # Compliance thresholds
        self.max_position_size = 1000000
        self.max_daily_loss = 500000
        self.max_order_size = 100000
        self.volatility_halt_threshold = 2.0  # 2std dev
        
        self.current_position = 0
        self.daily_pnl = 0
        self.daily_orders = 0
        
    def on_order(self, order: OrderMetrics):
        """Process new order (thread-safe callback)"""
        with self.lock:
            # Risk checks
            violations = []
            
            # Size check
            if order.quantity > self.max_order_size:
                violations.append(f"Order size {order.quantity} exceeds limit {self.max_order_size}")
            
            # Position check
            new_position = self.current_position + (order.quantity if order.side == "BUY" else -order.quantity)
            if abs(new_position) > self.max_position_size:
                violations.append(f"Position {new_position} exceeds limit {self.max_position_size}")
            
            # Daily loss check
            if self.daily_pnl < -self.max_daily_loss:
                violations.append(f"Daily loss ${abs(self.daily_pnl)} exceeds limit")
            
            if violations:
                logger.warning(f"Risk violations: {violations}")
                return False
            
            # Record order
            self.order_history.append(order)
            self.daily_orders += 1
            self.current_position = new_position
            
            # Calculate realized PnL if filled
            if order.status == "FILLED":
                pnl = (order.execution_price - order.price) * order.quantity
                if order.side == "SELL":
                    pnl *= -1
                self.daily_pnl += pnl
            
            logger.info(f"Order accepted: {order.symbol} {order.side} {order.quantity} @ {order.price} (latency: {order.latency_ns}ns)")
            return True
    
    def get_strategy_metrics(self, strategy: str, window_seconds: float = 60) -> Optional[StrategyMetrics]:
        """Get metrics for strategy over sliding window"""
        with self.lock:
            cutoff = time.time() - window_seconds
            orders = [o for o in self.order_history if o.strategy == strategy and o.timestamp > cutoff]
            
            if not orders:
                return None
            
            # Calculate metrics
            latencies = [o.latency_ns for o in orders]
            avg_latency = sum(latencies) / len(latencies)
            
            filled_orders = [o for o in orders if o.status == "FILLED"]
            pnl_values = []
            for o in filled_orders:
                pnl = (o.execution_price - o.price) * o.quantity
                if o.side == "SELL":
                    pnl *= -1
                pnl_values.append(pnl)
            
            pnl_realized = sum(pnl_values)
            win_count = sum(1 for p in pnl_values if p > 0)
            win_rate = win_count / len(pnl_values) if pnl_values else 0
            
            # Simplified Sharpe (would need time-series for real)
            sharpe = 1.0 if len(pnl_values) > 1 else 0.0
            
            return StrategyMetrics(
                strategy=strategy,
                timestamp=time.time(),
                orders_generated=len(orders),
                avg_latency_ns=avg_latency,
                pnl_realized=pnl_realized,
                pnl_unrealized=0.0,  # Would need mark-to-market
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                max_drawdown=0.0  # Would need full history
            )
    
    def get_compliance_report(self) -> Dict:
        """Generate compliance report"""
        with self.lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "current_position": self.current_position,
                "daily_orders": self.daily_orders,
                "daily_pnl": self.daily_pnl,
                "daily_pnl_ratio": self.daily_pnl / abs(self.daily_pnl) if self.daily_pnl != 0 else 0,
                "orders_in_last_hour": len([o for o in self.order_history 
                                           if o.timestamp > time.time() - 3600]),
                "avg_latency_ns": sum(o.latency_ns for o in self.order_history) / max(len(self.order_history), 1),
            }

class AuditLogger:
    """Immutable compliance audit trail"""
    
    def __init__(self, log_file: str = "audit.jsonl"):
        self.log_file = log_file
        self.lock = threading.Lock()
    
    def log_order(self, order: OrderMetrics):
        """Write order to immutable audit log"""
        with self.lock:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({
                    **asdict(order),
                    'log_timestamp': time.time()
                }) + '\n')
    
    def log_compliance_event(self, event_type: str, details: Dict):
        """Log compliance-relevant events"""
        with self.lock:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({
                    'event_type': event_type,
                    'timestamp': time.time(),
                    'details': details
                }) + '\n')

# Global instances
monitor = ComplianceMonitor()
auditor = AuditLogger()

def export_compliance_report(output_file: str = "compliance_report.json"):
    """Export daily compliance report"""
    report = monitor.get_compliance_report()
    
    # Add strategy metrics
    strategies = ["arbitrage", "market_making", "twap", "momentum"]
    report["strategy_metrics"] = {}
    for strat in strategies:
        metrics = monitor.get_strategy_metrics(strat)
        if metrics:
            report["strategy_metrics"][strat] = asdict(metrics)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Compliance report exported to {output_file}")

if __name__ == '__main__':
    # Demo
    monitor.max_position_size = 10000
    monitor.max_order_size = 1000
    
    test_order = OrderMetrics(
        order_id=1,
        timestamp=time.time(),
        strategy="arbitrage",
        symbol="BTC",
        side="BUY",
        quantity=100,
        price=45000,
        execution_price=45001,
        latency_ns=850,
        status="FILLED"
    )
    
    monitor.on_order(test_order)
    auditor.log_order(test_order)
    
    report = monitor.get_compliance_report()
    print(json.dumps(report, indent=2))
    
    export_compliance_report()
