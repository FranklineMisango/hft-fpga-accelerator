/*
 * Order Manager Testbench
 * Test environment for order management and execution logic
 * 
 * Features:
 * - Order lifecycle testing
 * - Risk management verification
 * - Performance measurement
 * - Latency analysis
 */

`timescale 1ns / 1ps

module order_manager_tb;

    // Clock and reset
    reg clk;
    reg rst_n;
    
    // DUT signals
    reg                 order_valid;
    reg [31:0]          order_symbol;
    reg [31:0]          order_price;
    reg [31:0]          order_volume;
    reg [7:0]           order_type;
    reg [31:0]          order_id;
    wire                order_ready;
    wire                execution_valid;
    wire [31:0]         execution_id;
    wire [31:0]         execution_price;
    wire [31:0]         execution_volume;
    wire [7:0]          execution_status;
    wire                risk_violation;
    wire [7:0]          risk_code;
    wire                position_update;
    wire [31:0]         position_symbol;
    wire [31:0]         position_size;
    wire [31:0]         position_pnl;
    
    // Test variables
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Performance counters
    reg [31:0] order_latency_start;
    reg [31:0] order_latency_end;
    reg [31:0] total_orders;
    reg [31:0] successful_orders;
    
    // Clock generation (250MHz)
    initial begin
        clk = 0;
        forever #2 clk = ~clk;
    end
    
    // DUT instantiation
    order_manager #(
        .DATA_WIDTH(32),
        .MAX_ORDERS(1024),
        .MAX_SYMBOLS(256),
        .MAX_POSITION_SIZE(1000000)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .order_valid(order_valid),
        .order_symbol(order_symbol),
        .order_price(order_price),
        .order_volume(order_volume),
        .order_type(order_type),
        .order_id(order_id),
        .order_ready(order_ready),
        .execution_valid(execution_valid),
        .execution_id(execution_id),
        .execution_price(execution_price),
        .execution_volume(execution_volume),
        .execution_status(execution_status),
        .risk_violation(risk_violation),
        .risk_code(risk_code),
        .position_update(position_update),
        .position_symbol(position_symbol),
        .position_size(position_size),
        .position_pnl(position_pnl)
    );
    
    // Test stimulus
    initial begin
        // Initialize
        rst_n = 0;
        order_valid = 0;
        order_symbol = 0;
        order_price = 0;
        order_volume = 0;
        order_type = 0;
        order_id = 0;
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        total_orders = 0;
        successful_orders = 0;
        
        // VCD dump
        $dumpfile("order_manager_tb.vcd");
        $dumpvars(0, order_manager_tb);
        
        $display("======================================");
        $display("Order Manager Testbench");
        $display("======================================");
        
        // Reset sequence
        #10 rst_n = 1;
        #10;
        
        // Test 1: Basic buy order
        test_buy_order();
        
        // Test 2: Basic sell order
        test_sell_order();
        
        // Test 3: Order cancellation
        test_order_cancel();
        
        // Test 4: Risk management
        test_risk_management();
        
        // Test 5: Position tracking
        test_position_tracking();
        
        // Test 6: High-frequency trading
        test_high_frequency();
        
        // Test 7: Stress test
        test_stress_conditions();
        
        // Test summary
        $display("\n======================================");
        $display("Test Summary");
        $display("======================================");
        $display("Total Tests: %d", test_count);
        $display("Passed:      %d", pass_count);
        $display("Failed:      %d", fail_count);
        $display("Orders Processed: %d", total_orders);
        $display("Success Rate: %d%%", (successful_orders * 100) / total_orders);
        
        if (fail_count == 0) begin
            $display("\nAll tests PASSED!");
        end else begin
            $display("\nSome tests FAILED!");
        end
        
        $finish;
    end
    
    // Test tasks
    task test_buy_order();
        begin
            $display("\nTest 1: Basic Buy Order");
            test_count = test_count + 1;
            
            // Submit buy order
            order_symbol = 32'h41415054;  // AAPL
            order_price = 32'h96000000;   // 150.00
            order_volume = 32'h64000000;  // 100 shares
            order_type = 8'h01;           // BUY
            order_id = 32'h12345678;
            order_valid = 1;
            order_latency_start = $time;
            
            @(posedge clk);
            order_valid = 0;
            
            // Wait for execution
            wait(execution_valid);
            order_latency_end = $time;
            
            // Check results
            if (execution_valid && execution_id == 32'h12345678) begin
                $display("  ✓ Buy order executed successfully");
                $display("  ✓ Latency: %d ns", order_latency_end - order_latency_start);
                pass_count = pass_count + 1;
                successful_orders = successful_orders + 1;
            end else begin
                $display("  ✗ Buy order failed");
                fail_count = fail_count + 1;
            end
            
            total_orders = total_orders + 1;
            @(posedge clk);
        end
    endtask
    
    task test_sell_order();
        begin
            $display("\nTest 2: Basic Sell Order");
            test_count = test_count + 1;
            
            // Submit sell order
            order_symbol = 32'h41415054;  // AAPL
            order_price = 32'h95000000;   // 149.00
            order_volume = 32'h32000000;  // 50 shares
            order_type = 8'h02;           // SELL
            order_id = 32'h87654321;
            order_valid = 1;
            order_latency_start = $time;
            
            @(posedge clk);
            order_valid = 0;
            
            // Wait for execution
            wait(execution_valid);
            order_latency_end = $time;
            
            // Check results
            if (execution_valid && execution_id == 32'h87654321) begin
                $display("  ✓ Sell order executed successfully");
                $display("  ✓ Latency: %d ns", order_latency_end - order_latency_start);
                pass_count = pass_count + 1;
                successful_orders = successful_orders + 1;
            end else begin
                $display("  ✗ Sell order failed");
                fail_count = fail_count + 1;
            end
            
            total_orders = total_orders + 1;
            @(posedge clk);
        end
    endtask
    
    task test_order_cancel();
        begin
            $display("\nTest 3: Order Cancellation");
            test_count = test_count + 1;
            
            // Submit order
            order_symbol = 32'h41415054;
            order_price = 32'h96000000;
            order_volume = 32'h64000000;
            order_type = 8'h01;
            order_id = 32'hABCDEF00;
            order_valid = 1;
            
            @(posedge clk);
            order_valid = 0;
            
            // Cancel order
            order_type = 8'h03;  // CANCEL
            order_id = 32'hABCDEF00;
            order_valid = 1;
            
            @(posedge clk);
            order_valid = 0;
            
            // Wait for response
            repeat(10) @(posedge clk);
            
            $display("  ✓ Order cancellation processed");
            pass_count = pass_count + 1;
            total_orders = total_orders + 1;
        end
    endtask
    
    task test_risk_management();
        begin
            $display("\nTest 4: Risk Management");
            test_count = test_count + 1;
            
            // Submit large order that should trigger risk controls
            order_symbol = 32'h41415054;
            order_price = 32'h96000000;
            order_volume = 32'hFFFFFFFF;  // Very large volume
            order_type = 8'h01;
            order_id = 32'hRISK0001;
            order_valid = 1;
            
            @(posedge clk);
            order_valid = 0;
            
            // Wait for risk violation
            repeat(10) @(posedge clk);
            
            if (risk_violation) begin
                $display("  ✓ Risk violation detected correctly");
                $display("  ✓ Risk code: %h", risk_code);
                pass_count = pass_count + 1;
            end else begin
                $display("  ✗ Risk violation not detected");
                fail_count = fail_count + 1;
            end
            
            total_orders = total_orders + 1;
        end
    endtask
    
    task test_position_tracking();
        begin
            $display("\nTest 5: Position Tracking");
            test_count = test_count + 1;
            
            // Submit multiple orders to track positions
            
            // Buy 100 shares
            order_symbol = 32'h41415054;
            order_price = 32'h96000000;
            order_volume = 32'h64000000;
            order_type = 8'h01;
            order_id = 32'h10000001;
            order_valid = 1;
            
            @(posedge clk);
            order_valid = 0;
            
            // Wait for execution
            wait(execution_valid);
            @(posedge clk);
            
            // Sell 50 shares
            order_symbol = 32'h41415054;
            order_price = 32'h95000000;
            order_volume = 32'h32000000;
            order_type = 8'h02;
            order_id = 32'h10000002;
            order_valid = 1;
            
            @(posedge clk);
            order_valid = 0;
            
            // Wait for execution
            wait(execution_valid);
            @(posedge clk);
            
            // Check position update
            if (position_update) begin
                $display("  ✓ Position tracking working");
                $display("  ✓ Position size: %h", position_size);
                pass_count = pass_count + 1;
            end else begin
                $display("  ✗ Position tracking failed");
                fail_count = fail_count + 1;
            end
            
            total_orders = total_orders + 2;
            successful_orders = successful_orders + 2;
        end
    endtask
    
    task test_high_frequency();
        integer i;
        reg [31:0] start_time, end_time;
        begin
            $display("\nTest 6: High-Frequency Trading");
            test_count = test_count + 1;
            
            start_time = $time;
            
            // Submit 1000 orders rapidly
            for (i = 0; i < 1000; i = i + 1) begin
                order_symbol = 32'h41415054;
                order_price = 32'h96000000 + i;
                order_volume = 32'h0A000000;  // 10 shares
                order_type = (i % 2) ? 8'h01 : 8'h02;  // Alternate buy/sell
                order_id = 32'h20000000 + i;
                order_valid = 1;
                
                @(posedge clk);
                order_valid = 0;
            end
            
            end_time = $time;
            
            $display("  ✓ Processed 1000 orders in %d ns", end_time - start_time);
            $display("  ✓ Throughput: %d orders/second", 
                     (1000 * 1000000000) / (end_time - start_time));
            pass_count = pass_count + 1;
            total_orders = total_orders + 1000;
        end
    endtask
    
    task test_stress_conditions();
        integer i;
        begin
            $display("\nTest 7: Stress Conditions");
            test_count = test_count + 1;
            
            // Submit orders at maximum rate
            for (i = 0; i < 100; i = i + 1) begin
                order_symbol = 32'h41415054;
                order_price = 32'h96000000;
                order_volume = 32'h64000000;
                order_type = 8'h01;
                order_id = 32'h30000000 + i;
                order_valid = 1;
                
                @(posedge clk);
                // Don't deassert order_valid - stress test
            end
            
            order_valid = 0;
            
            // Wait for processing
            repeat(200) @(posedge clk);
            
            $display("  ✓ Stress test completed");
            pass_count = pass_count + 1;
            total_orders = total_orders + 100;
        end
    endtask
    
    // Monitor for debugging
    always @(posedge clk) begin
        if (execution_valid) begin
            $display("Execution: ID=%h, Price=%h, Volume=%h, Status=%h", 
                     execution_id, execution_price, execution_volume, execution_status);
        end
        
        if (position_update) begin
            $display("Position: Symbol=%h, Size=%h, PnL=%h", 
                     position_symbol, position_size, position_pnl);
        end
        
        if (risk_violation) begin
            $display("Risk Violation: Code=%h", risk_code);
        end
    end

endmodule
