`timescale 1ns/1ps

module order_book_tb;

    localparam PRICE_WIDTH  = 32;
    localparam VOLUME_WIDTH = 32;
    localparam SYMBOL_WIDTH = 32;
    localparam DEPTH        = 8;

    reg clk, rst_n;
    reg update_valid;
    reg [SYMBOL_WIDTH-1:0] update_symbol;
    reg [PRICE_WIDTH-1:0]  update_price;
    reg [VOLUME_WIDTH-1:0] update_volume;
    reg                    update_side;
    reg [1:0]              update_action;
    reg [2:0]              query_level;
    reg                    query_side;

    wire [PRICE_WIDTH-1:0]  best_bid, best_ask, mid_price, spread;
    wire [VOLUME_WIDTH-1:0] best_bid_qty, best_ask_qty;
    wire                    book_valid;
    wire [PRICE_WIDTH-1:0]  query_price;
    wire [VOLUME_WIDTH-1:0] query_volume;
    wire [31:0]             update_count;
    wire [15:0]             bid_levels, ask_levels;

    order_book #(
        .PRICE_WIDTH(PRICE_WIDTH),
        .VOLUME_WIDTH(VOLUME_WIDTH),
        .SYMBOL_WIDTH(SYMBOL_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .update_valid(update_valid),
        .update_symbol(update_symbol),
        .update_price(update_price),
        .update_volume(update_volume),
        .update_side(update_side),
        .update_action(update_action),
        .best_bid(best_bid), .best_bid_qty(best_bid_qty),
        .best_ask(best_ask), .best_ask_qty(best_ask_qty),
        .book_valid(book_valid),
        .mid_price(mid_price), .spread(spread),
        .query_level(query_level), .query_side(query_side),
        .query_price(query_price), .query_volume(query_volume),
        .update_count(update_count),
        .bid_levels(bid_levels), .ask_levels(ask_levels)
    );

    always #5 clk = ~clk;

    task book_update;
        input [PRICE_WIDTH-1:0]  price;
        input [VOLUME_WIDTH-1:0] vol;
        input                    side;
        input [1:0]              action;
        begin
            update_valid  = 1'b1;
            update_symbol = 32'hDEAD_BEEF;
            update_price  = price;
            update_volume = vol;
            update_side   = side;
            update_action = action;
            @(posedge clk); #1;
            update_valid  = 1'b0;
            @(posedge clk); #1;
        end
    endtask

    integer pass_count, fail_count;

    initial begin
        clk = 0; rst_n = 0; pass_count = 0; fail_count = 0;
        update_valid = 0; query_level = 0; query_side = 0;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        // Test 1: Add bid levels
        book_update(32'd1000, 32'd500, 1'b0, 2'b00); // bid @ 1000
        book_update(32'd999,  32'd300, 1'b0, 2'b00); // bid @ 999
        book_update(32'd1001, 32'd200, 1'b1, 2'b00); // ask @ 1001
        book_update(32'd1002, 32'd100, 1'b1, 2'b00); // ask @ 1002
        repeat(2) @(posedge clk); #1;

        if (book_valid) begin
            $display("PASS: Book valid after inserts"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Book not valid"); fail_count = fail_count + 1;
        end

        if (best_bid == 32'd1000) begin
            $display("PASS: Best bid = %0d", best_bid); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected best bid 1000, got %0d", best_bid); fail_count = fail_count + 1;
        end

        if (best_ask == 32'd1001) begin
            $display("PASS: Best ask = %0d", best_ask); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected best ask 1001, got %0d", best_ask); fail_count = fail_count + 1;
        end

        // Test 2: Spread = ask - bid = 1
        if (spread == 32'd1) begin
            $display("PASS: Spread = %0d", spread); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected spread 1, got %0d", spread); fail_count = fail_count + 1;
        end

        // Test 3: Delete best bid, next level becomes best
        book_update(32'd1000, 32'd0, 1'b0, 2'b01); // delete bid @ 1000
        repeat(2) @(posedge clk); #1;
        if (best_bid == 32'd999) begin
            $display("PASS: Best bid after delete = %0d", best_bid); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected best bid 999 after delete, got %0d", best_bid); fail_count = fail_count + 1;
        end

        // Test 4: Mid price
        if (mid_price == ((32'd999 + 32'd1001) >> 1)) begin
            $display("PASS: Mid price = %0d", mid_price); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Unexpected mid price %0d", mid_price); fail_count = fail_count + 1;
        end

        $display("---");
        $display("order_book_tb: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("PASS: All order_book tests passed");
        else                 $display("FAIL: Some order_book tests failed");
        $finish;
    end

endmodule
