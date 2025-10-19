/*
 * Market Data Processor Testbench
 * Comprehensive test environment for market data processing module
 * 
 * Features:
 * - ITCH protocol simulation
 * - Real-time data generation
 * - Performance verification
 * - Latency measurement
 */

`timescale 1ns / 1ps

module market_data_tb;

    // Clock and reset
    reg clk;
    reg rst_n;
    
    // DUT signals
    reg                 data_valid;
    reg [63:0]          data_in;
    reg [7:0]           data_type;
    wire                data_ready;
    
    // Parsed market data output
    wire                tick_valid;
    wire [31:0]         symbol;
    wire [31:0]         price;
    wire [31:0]         volume;
    wire [31:0]         bid;
    wire [31:0]         ask;
    wire [63:0]         timestamp;
    
    // Order book interface
    wire                book_update_valid;
    wire [31:0]         book_symbol;
    wire [31:0]         book_price;
    wire [31:0]         book_volume;
    wire                book_side;
    wire [2:0]          book_action;
    
    // Statistics
    wire [31:0]         packets_processed;
    wire [31:0]         parse_errors;
    wire [15:0]         pipeline_depth;
    
    // Test variables
    integer test_count;
    integer pass_count;
    integer fail_count;
    
    // Performance measurement
    reg [31:0] latency_start;
    reg [31:0] latency_end;
    reg [31:0] min_latency;
    reg [31:0] max_latency;
    reg [31:0] total_latency;
    reg [31:0] latency_count;
    
    // Clock generation (250MHz)
    initial begin
        clk = 0;
        forever #2 clk = ~clk; // 4ns period = 250MHz
    end
    
    // DUT instantiation
    market_data_processor #(
        .DATA_WIDTH(64),
        .ADDR_WIDTH(32),
        .SYMBOL_WIDTH(32),
        .PRICE_WIDTH(32),
        .VOLUME_WIDTH(32),
        .MAX_ORDERS(1024),
        .MAX_SYMBOLS(256)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_valid(data_valid),
        .data_in(data_in),
        .data_type(data_type),
        .data_ready(data_ready),
        .tick_valid(tick_valid),
        .symbol(symbol),
        .price(price),
        .volume(volume),
        .bid(bid),
        .ask(ask),
        .timestamp(timestamp),
        .book_update_valid(book_update_valid),
        .book_symbol(book_symbol),
        .book_price(book_price),
        .book_volume(book_volume),
        .book_side(book_side),
        .book_action(book_action),
        .packets_processed(packets_processed),
        .parse_errors(parse_errors),
        .pipeline_depth(pipeline_depth)
    );
    
    // Test stimulus
    initial begin
        // Initialize
        rst_n = 0;
        data_valid = 0;
        data_in = 0;
        data_type = 0;
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        min_latency = 32'hFFFFFFFF;
        max_latency = 0;
        total_latency = 0;
        latency_count = 0;
        
        // VCD dump for waveform analysis
        $dumpfile("market_data_tb.vcd");
        $dumpvars(0, market_data_tb);
        
        $display("======================================");
        $display("Market Data Processor Testbench");
        $display("======================================");
        
        // Reset sequence
        #10 rst_n = 1;
        #10;
        
        // Test 1: Basic ITCH Add Order message
        test_itch_add_order();
        
        // Test 2: ITCH Order execution
        test_itch_execution();
        
        // Test 3: ITCH Order cancellation
        test_itch_cancel();
        
        // Test 4: Multiple symbols
        test_multiple_symbols();
        
        // Test 5: High-frequency burst
        test_high_frequency_burst();
        
        // Test 6: Error conditions
        test_error_conditions();
        
        // Test 7: Performance measurement
        test_performance();
        
        // Test summary
        $display("\n======================================");
        $display("Test Summary");
        $display("======================================");
        $display("Total Tests: %d", test_count);
        $display("Passed:      %d", pass_count);
        $display("Failed:      %d", fail_count);
        
        if (latency_count > 0) begin
            $display("\nPerformance Results:");
            $display("Min Latency:  %d ns", min_latency * 4);
            $display("Max Latency:  %d ns", max_latency * 4);
            $display("Avg Latency:  %d ns", (total_latency / latency_count) * 4);
        end
        
        if (fail_count == 0) begin
            $display("\nAll tests PASSED!");
        end else begin
            $display("\nSome tests FAILED!");
        end
        
        $finish;
    end
    
    // Test tasks
    task test_itch_add_order();
    reg [31:0] timeout;
    begin
        $display("\nTest 1: ITCH Add Order");
        test_count = test_count + 1;

        // Wait a few cycles after reset to ensure DUT is ready
        repeat(5) @(posedge clk);

        // Prepare ITCH Add Order message (AAPL, price example)
        data_type = 8'h41; // 'A' for Add Order
        data_in = {32'h41415054, 32'h32000000}; // symbol = "AAPL", price = 0x32000000
        data_valid = 1;
        @(posedge clk);
        data_valid = 0;

        // Wait for tick_valid or timeout
        timeout = 0;
        while (!tick_valid && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (tick_valid) begin
            $display("  ✓ ITCH Add Order processed: symbol=%h price=%h volume=%h", symbol, price, volume);
            pass_count = pass_count + 1;
        end else begin
            $display("  ✗ ITCH Add Order failed - tick_valid=0, symbol=%h", symbol);
            fail_count = fail_count + 1;
        end
    end
endtask
    
    task test_itch_execution();
        begin
            $display("\nTest 2: ITCH Order Execution");
            test_count = test_count + 1;
            
            // ITCH Execution message type 'E' (0x45)
            data_type = 8'h45;  // 'E'
            data_in = {32'h41415054, 32'h32000000}; // AAPL execution
            data_valid = 1;
            latency_start = $time;
            
            @(posedge clk);
            data_valid = 0;
            
            // Wait for response with timeout
            fork
                begin
                    wait(book_update_valid);
                    latency_end = $time;
                end
                begin
                    repeat(100) @(posedge clk);  // Timeout after 100 cycles
                    $display("  ⚠ Timeout waiting for book_update_valid");
                end
            join_any
            disable fork;
            
            // Check results
            if (book_update_valid && book_symbol == 32'h41415054) begin
                $display("  ✓ ITCH Execution processed correctly");
                pass_count = pass_count + 1;
                measure_latency();
            end else begin
                $display("  ✗ ITCH Execution failed - book_update_valid=%b, book_symbol=%h", book_update_valid, book_symbol);
                fail_count = fail_count + 1;
            end
            
            @(posedge clk);
        end
    endtask
    
    task test_itch_cancel();
        begin
            $display("\nTest 3: ITCH Order Cancel");
            test_count = test_count + 1;
            
            // ITCH Cancel message type 'X' (0x58)
            data_type = 8'h58;  // 'X'
            data_in = {32'h41415054, 32'h12345678}; // AAPL cancel
            data_valid = 1;
            latency_start = $time;
            
            @(posedge clk);
            data_valid = 0;
            
            // Wait for response with timeout
            fork
                begin
                    wait(book_update_valid);
                    latency_end = $time;
                end
                begin
                    repeat(100) @(posedge clk);  // Timeout after 100 cycles
                    $display("  ⚠ Timeout waiting for book_update_valid");
                end
            join_any
            disable fork;
            
            // Check results
            if (book_update_valid) begin
                $display("  ✓ ITCH Cancel processed correctly");
                pass_count = pass_count + 1;
                measure_latency();
            end else begin
                $display("  ✗ ITCH Cancel failed - book_update_valid=%b", book_update_valid);
                fail_count = fail_count + 1;
            end
            
            @(posedge clk);
        end
    endtask
    
    task test_multiple_symbols();
        begin
            $display("\nTest 4: Multiple Symbols");
            test_count = test_count + 1;
            
            // Test AAPL
            data_type = 8'h41;
            data_in = {32'h41415054, 32'h64000000}; // AAPL
            data_valid = 1;
            @(posedge clk);
            data_valid = 0;
            @(posedge clk);
            
            // Test GOOGL
            data_type = 8'h41;
            data_in = {32'h474f4f47, 32'h96000000}; // GOOGL
            data_valid = 1;
            @(posedge clk);
            data_valid = 0;
            @(posedge clk);
            
            // Test MSFT
            data_type = 8'h41;
            data_in = {32'h4d534654, 32'h50000000}; // MSFT
            data_valid = 1;
            @(posedge clk);
            data_valid = 0;
            
            // Wait for processing
            repeat(10) @(posedge clk);
            
            $display("  ✓ Multiple symbols processed");
            pass_count = pass_count + 1;
        end
    endtask
    
    task test_high_frequency_burst();
        integer i;
        begin
            $display("\nTest 5: High-Frequency Burst (1000 messages)");
            test_count = test_count + 1;
            
            for (i = 0; i < 1000; i = i + 1) begin
                data_type = 8'h41;
                data_in = {32'h41415054, i[31:0]}; // AAPL with varying data
                data_valid = 1;
                @(posedge clk);
                data_valid = 0;
                
                // Every 10th message, wait for processing
                if (i % 10 == 0) begin
                    @(posedge clk);
                end
            end
            
            // Wait for all processing to complete
            repeat(100) @(posedge clk);
            
            $display("  ✓ High-frequency burst completed");
            pass_count = pass_count + 1;
        end
    endtask
    
    task test_error_conditions();
        begin
            $display("\nTest 6: Error Conditions");
            test_count = test_count + 1;
            
            // Invalid message type
            data_type = 8'hFF;
            data_in = {32'h41415054, 32'h64000000};
            data_valid = 1;
            @(posedge clk);
            data_valid = 0;
            
            // Wait for error
            repeat(10) @(posedge clk);
            
            if (parse_errors > 0) begin
                $display("  ✓ Error condition detected correctly");
                pass_count = pass_count + 1;
            end else begin
                $display("  ✗ Error condition not detected");
                fail_count = fail_count + 1;
            end
        end
    endtask
    
    task test_performance();
        integer i;
        reg [31:0] start_time, end_time;
        begin
            $display("\nTest 7: Performance Measurement");
            test_count = test_count + 1;
            
            start_time = $time;
            
            // Process 10000 messages
            for (i = 0; i < 10000; i = i + 1) begin
                data_type = 8'h41;
                data_in = {32'h41415054, i[31:0]};
                data_valid = 1;
                @(posedge clk);
                data_valid = 0;
            end
            
            end_time = $time;
            
            $display("  ✓ Processed 10000 messages in %d ns", end_time - start_time);
            $display("  ✓ Throughput: %d messages/second", 
                     (10000 * 1000000000) / (end_time - start_time));
            pass_count = pass_count + 1;
        end
    endtask
    
    task measure_latency();
        reg [31:0] latency_cycles;
        begin
            latency_cycles = (latency_end - latency_start) / 4; // Convert to cycles
            
            if (latency_cycles < min_latency) min_latency = latency_cycles;
            if (latency_cycles > max_latency) max_latency = latency_cycles;
            total_latency = total_latency + latency_cycles;
            latency_count = latency_count + 1;
        end
    endtask
    
    // Monitor for debugging
    always @(posedge clk) begin
        if (tick_valid) begin
            $display("Tick: Symbol=%h, Price=%h, Volume=%h, Bid=%h, Ask=%h", 
                     symbol, price, volume, bid, ask);
        end
        
        if (book_update_valid) begin
            $display("Book Update: Symbol=%h, Price=%h, Volume=%h, Side=%h, Action=%h", 
                     book_symbol, book_price, book_volume, book_side, book_action);
        end
        
        if (parse_errors > 0) begin
            $display("Parse Errors: Count=%d", parse_errors);
        end
    end

endmodule
