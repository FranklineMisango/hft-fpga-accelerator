`timescale 1ns/1ps

module pnl_tracker_tb;

    localparam PRICE_WIDTH  = 32;
    localparam VOLUME_WIDTH = 32;
    localparam SYMBOL_WIDTH = 32;

    reg clk, rst_n;
    reg fill_valid;
    reg [SYMBOL_WIDTH-1:0] fill_symbol;
    reg [PRICE_WIDTH-1:0]  fill_price;
    reg [VOLUME_WIDTH-1:0] fill_volume;
    reg                    fill_side;
    reg mark_valid;
    reg [SYMBOL_WIDTH-1:0] mark_symbol;
    reg [PRICE_WIDTH-1:0]  mark_price;
    reg [SYMBOL_WIDTH-1:0] query_symbol;

    wire signed [VOLUME_WIDTH-1:0] query_net_position;
    wire signed [PRICE_WIDTH-1:0]  query_unrealized_pnl;
    wire signed [PRICE_WIDTH-1:0]  query_realized_pnl;
    wire signed [PRICE_WIDTH-1:0]  total_realized_pnl;
    wire signed [PRICE_WIDTH-1:0]  total_unrealized_pnl;
    wire signed [VOLUME_WIDTH-1:0] net_position;
    wire signed [PRICE_WIDTH-1:0]  daily_pnl;

    pnl_tracker #(
        .PRICE_WIDTH(PRICE_WIDTH),
        .VOLUME_WIDTH(VOLUME_WIDTH),
        .SYMBOL_WIDTH(SYMBOL_WIDTH),
        .MAX_SYMBOLS(4)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .fill_valid(fill_valid), .fill_symbol(fill_symbol),
        .fill_price(fill_price), .fill_volume(fill_volume),
        .fill_side(fill_side),
        .mark_valid(mark_valid), .mark_symbol(mark_symbol),
        .mark_price(mark_price),
        .query_symbol(query_symbol),
        .query_net_position(query_net_position),
        .query_unrealized_pnl(query_unrealized_pnl),
        .query_realized_pnl(query_realized_pnl),
        .total_realized_pnl(total_realized_pnl),
        .total_unrealized_pnl(total_unrealized_pnl),
        .net_position(net_position),
        .daily_pnl(daily_pnl)
    );

    always #5 clk = ~clk;

    task do_fill;
        input [SYMBOL_WIDTH-1:0] sym;
        input [PRICE_WIDTH-1:0]  price;
        input [VOLUME_WIDTH-1:0] vol;
        input                    side;
        begin
            fill_valid = 1'b1; fill_symbol = sym;
            fill_price = price; fill_volume = vol; fill_side = side;
            @(posedge clk); #1;
            fill_valid = 1'b0;
            @(posedge clk); #1;
        end
    endtask

    task do_mark;
        input [SYMBOL_WIDTH-1:0] sym;
        input [PRICE_WIDTH-1:0]  price;
        begin
            mark_valid = 1'b1; mark_symbol = sym; mark_price = price;
            @(posedge clk); #1;
            mark_valid = 1'b0;
            @(posedge clk); #1;
        end
    endtask

    integer pass_count, fail_count;
    localparam SYM_A = 32'hAAAA_0001;

    initial begin
        clk = 0; rst_n = 0; pass_count = 0; fail_count = 0;
        fill_valid = 0; mark_valid = 0;
        fill_symbol = 0; fill_price = 0; fill_volume = 0; fill_side = 0;
        mark_symbol = 0; mark_price = 0;
        query_symbol = SYM_A;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        // Test 1: Buy 100 @ 1000 → position = +100
        do_fill(SYM_A, 32'd1000, 32'd100, 1'b0);
        if (query_net_position == 32'sd100) begin
            $display("PASS: Position after buy = %0d", query_net_position); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected position 100, got %0d", query_net_position); fail_count = fail_count + 1;
        end

        // Test 2: Mark up to 1100 → unrealized PnL = +10000
        do_mark(SYM_A, 32'd1100);
        repeat(2) @(posedge clk); #1;
        if (query_unrealized_pnl == 32'sd10000) begin
            $display("PASS: Unrealized PnL = %0d", query_unrealized_pnl); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected unrealized 10000, got %0d", query_unrealized_pnl); fail_count = fail_count + 1;
        end

        // Test 3: Sell 100 @ 1100 → realized PnL = +10000, position = 0
        do_fill(SYM_A, 32'd1100, 32'd100, 1'b1);
        if (query_net_position == 32'sd0) begin
            $display("PASS: Position flat after sell"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected position 0, got %0d", query_net_position); fail_count = fail_count + 1;
        end
        if (query_realized_pnl == 32'sd10000) begin
            $display("PASS: Realized PnL = %0d", query_realized_pnl); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected realized 10000, got %0d", query_realized_pnl); fail_count = fail_count + 1;
        end

        $display("---");
        $display("pnl_tracker_tb: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("PASS: All pnl_tracker tests passed");
        else                 $display("FAIL: Some pnl_tracker tests failed");
        $finish;
    end

endmodule
