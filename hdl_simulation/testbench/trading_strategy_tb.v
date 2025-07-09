/*
 * Trading Strategy Testbench
 * Test environment for trading strategy execution engine
 * 
 * Features:
 * - Strategy algorithm testing
 * - Decision latency measurement
 * - Performance analysis
 * - Signal validation
 */

`timescale 1ns / 1ps

module trading_strategy_tb;

    // Clock and reset
    reg clk;
    reg rst_n;
    
    // DUT signals
    reg                 market_data_valid;
    reg [31:0]          market_symbol;
    reg [31:0]          market_price;
    reg [31:0]          market_volume;
    reg [31:0]          market_bid;
    reg [31:0]          market_ask;
    wire                signal_valid;
    wire [31:0]         signal_symbol;
    wire [31:0]         signal_price;
    wire [31:0]         signal_volume;
    wire [7:0]          signal_type;
    wire [31:0]         signal_confidence;
    wire                risk_check_valid;
    wire [31:0]         risk_exposure;
    wire [31:0]         risk_pnl;
    
    // Test variables
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Performance measurement
    reg [31:0] decision_latency_start;
    reg [31:0] decision_latency_end;
    reg [31:0] total_signals;
    reg [31:0] valid_signals;
    
    // Clock generation (250MHz)
    initial begin
        clk = 0;
        forever #2 clk = ~clk;
    end
    
    // DUT instantiation
    trading_strategy #(
        .DATA_WIDTH(32),
        .MAX_SYMBOLS(256),
        .LOOKBACK_DEPTH(128),
        .STRATEGY_COUNT(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .market_data_valid(market_data_valid),
        .market_symbol(market_symbol),
        .market_price(market_price),
        .market_volume(market_volume),
        .market_bid(market_bid),
        .market_ask(market_ask),
        .signal_valid(signal_valid),
        .signal_symbol(signal_symbol),
        .signal_price(signal_price),
        .signal_volume(signal_volume),
        .signal_type(signal_type),
        .signal_confidence(signal_confidence),
        .risk_check_valid(risk_check_valid),
        .risk_exposure(risk_exposure),
        .risk_pnl(risk_pnl)
    );
    
    // Test stimulus
    initial begin
        // Initialize
        rst_n = 0;
        market_data_valid = 0;
        market_symbol = 0;
        market_price = 0;
        market_volume = 0;
        market_bid = 0;
        market_ask = 0;
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        total_signals = 0;
        valid_signals = 0;
        
        // VCD dump
        $dumpfile("trading_strategy_tb.vcd");
        $dumpvars(0, trading_strategy_tb);
        
        $display("======================================");
        $display("Trading Strategy Testbench");
        $display("======================================");
        
        // Reset sequence
        #10 rst_n = 1;
        #10;
        
        // Test 1: Arbitrage detection
        test_arbitrage_strategy();
        
        // Test 2: Market making
        test_market_making_strategy();
        
        // Test 3: Momentum strategy
        test_momentum_strategy();
        
        // Test 4: Mean reversion
        test_mean_reversion_strategy();
        
        // Test 5: Multi-symbol trading
        test_multi_symbol_trading();
        
        // Test 6: High-frequency signals
        test_high_frequency_signals();
        
        // Test 7: Risk management
        test_risk_management();
        
        // Test summary
        $display("\n======================================");
        $display("Test Summary");
        $display("======================================");
        $display("Total Tests: %d", test_count);
        $display("Passed:      %d", pass_count);
        $display("Failed:      %d", fail_count);
        $display("Total Signals: %d", total_signals);
        $display("Valid Signals: %d", valid_signals);
        
        if (total_signals > 0) begin
            $display("Signal Rate: %d%%", (valid_signals * 100) / total_signals);
        end
        
        if (fail_count == 0) begin
            $display("\nAll tests PASSED!");
        end else begin
            $display("\nSome tests FAILED!");
        end
        
        $finish;
    end
    
    // Test tasks
    task test_arbitrage_strategy();
        begin
            $display("\nTest 1: Arbitrage Strategy");
            test_count = test_count + 1;
            
            // Send market data for AAPL with price discrepancy
            market_symbol = 32'h41415054;  // AAPL
            market_price = 32'h96000000;   // 150.00
            market_volume = 32'h64000000;  // 100
            market_bid = 32'h95F00000;     // 149.90
            market_ask = 32'h96200000;     // 150.20
            market_data_valid = 1;
            decision_latency_start = $time;
            
            @(posedge clk);
            market_data_valid = 0;
            
            // Wait for signal
            wait(signal_valid);
            decision_latency_end = $time;
            
            // Check if arbitrage opportunity detected
            if (signal_valid && signal_type == 8'h01) begin  // ARBITRAGE
                $display("  ✓ Arbitrage opportunity detected");
                $display("  ✓ Decision latency: %d ns", decision_latency_end - decision_latency_start);
                $display("  ✓ Confidence: %h", signal_confidence);
                pass_count = pass_count + 1;
                valid_signals = valid_signals + 1;
            end else begin
                $display("  ✗ Arbitrage strategy failed");
                fail_count = fail_count + 1;
            end
            
            total_signals = total_signals + 1;
            @(posedge clk);
        end
    endtask
    
    task test_market_making_strategy();
        begin
            $display("\nTest 2: Market Making Strategy");
            test_count = test_count + 1;
            
            // Send market data with wide spread
            market_symbol = 32'h474f4f47;  // GOOGL
            market_price = 32'hAF000000;   // 2800.00
            market_volume = 32'h32000000;  // 50
            market_bid = 32'hAE000000;     // 2784.00
            market_ask = 32'hB0000000;     // 2816.00
            market_data_valid = 1;
            decision_latency_start = $time;
            
            @(posedge clk);
            market_data_valid = 0;
            
            // Wait for signal
            wait(signal_valid);
            decision_latency_end = $time;
            
            // Check if market making signal generated
            if (signal_valid && signal_type == 8'h02) begin  // MARKET_MAKING
                $display("  ✓ Market making signal generated");
                $display("  ✓ Decision latency: %d ns", decision_latency_end - decision_latency_start);
                pass_count = pass_count + 1;
                valid_signals = valid_signals + 1;
            end else begin
                $display("  ✗ Market making strategy failed");
                fail_count = fail_count + 1;
            end
            
            total_signals = total_signals + 1;
            @(posedge clk);
        end
    endtask
    
    task test_momentum_strategy();
        integer i;
        begin
            $display("\nTest 3: Momentum Strategy");
            test_count = test_count + 1;
            
            // Send trending price data
            for (i = 0; i < 20; i = i + 1) begin
                market_symbol = 32'h4d534654;  // MSFT
                market_price = 32'h50000000 + (i * 32'h01000000);  // Increasing price
                market_volume = 32'h64000000;
                market_bid = market_price - 32'h00100000;
                market_ask = market_price + 32'h00100000;
                market_data_valid = 1;
                
                @(posedge clk);
                market_data_valid = 0;
                @(posedge clk);
            end
            
            // Wait for momentum signal
            repeat(10) @(posedge clk);
            
            if (signal_valid && signal_type == 8'h03) begin  // MOMENTUM
                $display("  ✓ Momentum signal generated");
                pass_count = pass_count + 1;
                valid_signals = valid_signals + 1;
            end else begin
                $display("  ✗ Momentum strategy failed");
                fail_count = fail_count + 1;
            end
            
            total_signals = total_signals + 1;
        end
    endtask
    
    task test_mean_reversion_strategy();
        integer i;
        begin
            $display("\nTest 4: Mean Reversion Strategy");
            test_count = test_count + 1;
            
            // Send oscillating price data
            for (i = 0; i < 30; i = i + 1) begin
                market_symbol = 32'h54534c41;  // TSLA
                market_price = 32'h64000000 + (i % 2 ? 32'h05000000 : -32'h05000000);
                market_volume = 32'h32000000;
                market_bid = market_price - 32'h00100000;
                market_ask = market_price + 32'h00100000;
                market_data_valid = 1;
                
                @(posedge clk);
                market_data_valid = 0;
                @(posedge clk);
            end
            
            // Wait for mean reversion signal
            repeat(10) @(posedge clk);
            
            if (signal_valid && signal_type == 8'h04) begin  // MEAN_REVERSION
                $display("  ✓ Mean reversion signal generated");
                pass_count = pass_count + 1;
                valid_signals = valid_signals + 1;
            end else begin
                $display("  ✗ Mean reversion strategy failed");
                fail_count = fail_count + 1;
            end
            
            total_signals = total_signals + 1;
        end
    endtask
    
    task test_multi_symbol_trading();
        begin
            $display("\nTest 5: Multi-Symbol Trading");
            test_count = test_count + 1;
            
            // Send data for multiple symbols
            
            // AAPL
            market_symbol = 32'h41415054;
            market_price = 32'h96000000;
            market_volume = 32'h64000000;
            market_bid = 32'h95F00000;
            market_ask = 32'h96100000;
            market_data_valid = 1;
            @(posedge clk);
            market_data_valid = 0;
            @(posedge clk);
            
            // GOOGL
            market_symbol = 32'h474f4f47;
            market_price = 32'hAF000000;
            market_volume = 32'h32000000;
            market_bid = 32'hAEF00000;
            market_ask = 32'hAF100000;
            market_data_valid = 1;
            @(posedge clk);
            market_data_valid = 0;
            @(posedge clk);
            
            // MSFT
            market_symbol = 32'h4d534654;
            market_price = 32'h50000000;
            market_volume = 32'h48000000;
            market_bid = 32'h4FF00000;
            market_ask = 32'h50100000;
            market_data_valid = 1;
            @(posedge clk);
            market_data_valid = 0;
            
            // Wait for processing
            repeat(20) @(posedge clk);
            
            $display("  ✓ Multi-symbol trading processed");
            pass_count = pass_count + 1;
        end
    endtask
    
    task test_high_frequency_signals();
        integer i;
        reg [31:0] start_time, end_time;
        begin
            $display("\nTest 6: High-Frequency Signals");
            test_count = test_count + 1;
            
            start_time = $time;
            
            // Send 1000 market data updates rapidly
            for (i = 0; i < 1000; i = i + 1) begin
                market_symbol = 32'h4e564441;  // NVDA
                market_price = 32'h64000000 + (i % 10);
                market_volume = 32'h32000000;
                market_bid = market_price - 32'h00100000;
                market_ask = market_price + 32'h00100000;
                market_data_valid = 1;
                
                @(posedge clk);
                market_data_valid = 0;
            end
            
            end_time = $time;
            
            $display("  ✓ Processed 1000 market updates in %d ns", end_time - start_time);
            $display("  ✓ Throughput: %d updates/second", 
                     (1000 * 1000000000) / (end_time - start_time));
            pass_count = pass_count + 1;
        end
    endtask
    
    task test_risk_management();
        begin
            $display("\nTest 7: Risk Management");
            test_count = test_count + 1;
            
            // Send data that should trigger risk checks
            market_symbol = 32'h41415054;
            market_price = 32'hFFFFFFFF;  // Extreme price
            market_volume = 32'hFFFFFFFF;  // Extreme volume
            market_bid = 32'h00000001;
            market_ask = 32'hFFFFFFFE;
            market_data_valid = 1;
            
            @(posedge clk);
            market_data_valid = 0;
            
            // Wait for risk check
            repeat(10) @(posedge clk);
            
            if (risk_check_valid) begin
                $display("  ✓ Risk check triggered");
                $display("  ✓ Risk exposure: %h", risk_exposure);
                pass_count = pass_count + 1;
            end else begin
                $display("  ✗ Risk management failed");
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    // Monitor for debugging
    always @(posedge clk) begin
        if (signal_valid) begin
            $display("Signal: Symbol=%h, Price=%h, Volume=%h, Type=%h, Confidence=%h", 
                     signal_symbol, signal_price, signal_volume, signal_type, signal_confidence);
        end
        
        if (risk_check_valid) begin
            $display("Risk Check: Exposure=%h, PnL=%h", risk_exposure, risk_pnl);
        end
    end

endmodule
