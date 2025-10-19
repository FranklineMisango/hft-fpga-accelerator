// Testbench for HJB Calculator
`timescale 1ns/1ps

module hjb_calculator_tb;
    reg clk;
    reg rst_n;
    reg [63:0] mid_price;
    reg [31:0] inventory;
    reg [63:0] volatility;
    reg calculate_en;
    wire [63:0] optimal_bid;
    wire [63:0] optimal_ask;
    wire calculation_done;
    wire [31:0] latency_cycles;
    
    // Clock generation
    always #2 clk = ~clk; // 250MHz clock (4ns period)
    
    // DUT instantiation
    hjb_calculator dut (
        .clk(clk),
        .rst_n(rst_n),
        .mid_price(mid_price),
        .inventory(inventory),
        .volatility(volatility),
        .calculate_en(calculate_en),
        .optimal_bid(optimal_bid),
        .optimal_ask(optimal_ask),
        .calculation_done(calculation_done),
        .latency_cycles(latency_cycles)
    );
    
    // Test file I/O
    integer input_file, output_file;
    real mid_price_real, volatility_real;
    integer inventory_int;
    real bid_real, ask_real;
    
    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        calculate_en = 0;
        mid_price = 64'h0;
        inventory = 32'h0;
        volatility = 64'h0;
        
        // Reset
        #10 rst_n = 1;
        #10;
        
        // Try to read input from file
        input_file = $fopen("market_input.txt", "r");
        if (input_file) begin
            if ($fscanf(input_file, "%f,%d,%f", mid_price_real, inventory_int, volatility_real) == 3) begin
                mid_price = $realtobits(mid_price_real);
                inventory = inventory_int;
                volatility = $realtobits(volatility_real);
                $display("Read input: mid_price=%f, inventory=%d, volatility=%f", 
                         mid_price_real, inventory_int, volatility_real);
            end else begin
                // Default test values
                mid_price = $realtobits(100000.0);  // $100k BTC
                inventory = 5;                       // 5 BTC inventory
                volatility = $realtobits(0.3);       // 30% volatility
                $display("Using default test values");
            end
            $fclose(input_file);
        end else begin
            // Default test values if no input file
            mid_price = $realtobits(100000.0);
            inventory = 5;
            volatility = $realtobits(0.3);
            $display("No input file, using default values");
        end
        
        // Start calculation
        calculate_en = 1;
        #10;
        
        // Wait for completion
        wait(calculation_done);
        
        // Convert results to real numbers
        bid_real = $bitstoreal(optimal_bid);
        ask_real = $bitstoreal(optimal_ask);
        
        $display("HJB Calculation Results:");
        $display("Optimal Bid: %f", bid_real);
        $display("Optimal Ask: %f", ask_real);
        $display("Latency: %d cycles (%d ns)", latency_cycles, latency_cycles * 4);
        
        // Write output to file
        output_file = $fopen("strategy_output.txt", "w");
        if (output_file) begin
            $fwrite(output_file, "%f,%f\n", bid_real, ask_real);
            $fclose(output_file);
            $display("Results written to strategy_output.txt");
        end
        
        calculate_en = 0;
        #20;
        
        $display("Simulation completed successfully");
        $finish;
    end
    
    // VCD dump for waveform analysis
    initial begin
        $dumpfile("hjb_calculator.vcd");
        $dumpvars(0, hjb_calculator_tb);
    end
    
    // Timeout
    initial begin
        #1000;
        $display("ERROR: Simulation timeout");
        $finish;
    end

endmodule