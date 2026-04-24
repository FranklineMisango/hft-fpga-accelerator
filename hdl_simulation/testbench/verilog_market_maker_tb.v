`timescale 1ns/1ps

module verilog_market_maker_tb;
    reg clk;
    reg rst_n;
    reg calculate_en;

    reg signed [31:0] mid_ticks;
    reg signed [31:0] best_bid_ticks;
    reg signed [31:0] best_ask_ticks;
    reg signed [31:0] inventory_milli;
    reg signed [31:0] volatility_bp;
    reg signed [31:0] bid_qty_milli;
    reg signed [31:0] ask_qty_milli;

    wire signed [31:0] quote_bid_ticks;
    wire signed [31:0] quote_ask_ticks;
    wire replace_hint;
    wire cancel_hint;
    wire done;
    wire [31:0] latency_cycles;

    integer input_file;
    integer output_file;
    integer read_count;
    integer v_mid;
    integer v_bid;
    integer v_ask;
    integer v_inv;
    integer v_vol;
    integer v_bid_qty;
    integer v_ask_qty;

    verilog_market_maker dut (
        .clk(clk),
        .rst_n(rst_n),
        .calculate_en(calculate_en),
        .mid_ticks(mid_ticks),
        .best_bid_ticks(best_bid_ticks),
        .best_ask_ticks(best_ask_ticks),
        .inventory_milli(inventory_milli),
        .volatility_bp(volatility_bp),
        .bid_qty_milli(bid_qty_milli),
        .ask_qty_milli(ask_qty_milli),
        .quote_bid_ticks(quote_bid_ticks),
        .quote_ask_ticks(quote_ask_ticks),
        .replace_hint(replace_hint),
        .cancel_hint(cancel_hint),
        .done(done),
        .latency_cycles(latency_cycles)
    );

    always #2 clk = ~clk;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        calculate_en = 1'b0;
        mid_ticks = 32'sd0;
        best_bid_ticks = 32'sd0;
        best_ask_ticks = 32'sd0;
        inventory_milli = 32'sd0;
        volatility_bp = 32'sd0;
        bid_qty_milli = 32'sd0;
        ask_qty_milli = 32'sd0;

        #10 rst_n = 1'b1;
        #10;

        input_file = $fopen("market_input_verilog.txt", "r");
        if (input_file) begin
            read_count = $fscanf(
                input_file,
                "%d,%d,%d,%d,%d,%d,%d",
                v_mid,
                v_bid,
                v_ask,
                v_inv,
                v_vol,
                v_bid_qty,
                v_ask_qty
            );
            $fclose(input_file);
            if (read_count != 7) begin
                v_mid = 10000000;
                v_bid = 9999950;
                v_ask = 10000050;
                v_inv = 0;
                v_vol = 30;
                v_bid_qty = 1200;
                v_ask_qty = 1300;
            end
        end else begin
            v_mid = 10000000;
            v_bid = 9999950;
            v_ask = 10000050;
            v_inv = 0;
            v_vol = 30;
            v_bid_qty = 1200;
            v_ask_qty = 1300;
        end

        mid_ticks = v_mid;
        best_bid_ticks = v_bid;
        best_ask_ticks = v_ask;
        inventory_milli = v_inv;
        volatility_bp = v_vol;
        bid_qty_milli = v_bid_qty;
        ask_qty_milli = v_ask_qty;

        calculate_en = 1'b1;
        #10;
        calculate_en = 1'b0;

        wait(done);

        output_file = $fopen("strategy_verilog_output.txt", "w");
        if (output_file) begin
            $fwrite(
                output_file,
                "%0d,%0d,%0d,%0d,%0d\n",
                quote_bid_ticks,
                quote_ask_ticks,
                replace_hint ? 1 : 0,
                cancel_hint ? 1 : 0,
                latency_cycles
            );
            $fclose(output_file);
        end

        #20;
        $finish;
    end

    initial begin
        #2000;
        $display("ERROR: verilog_market_maker_tb timeout");
        $finish;
    end
endmodule
