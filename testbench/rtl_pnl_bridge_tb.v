`timescale 1ns/1ps
// Bridge harness for pnl_tracker.
// Reads one line from rtl_pnl_input.txt:
//   <fill_valid>,<fill_side>,<fill_price_cents>,<fill_volume_milli>,<mark_price_cents>
// Writes one line to rtl_pnl_output.txt:
//   <total_realized_pnl>,<total_unrealized_pnl>,<net_position>,<daily_pnl>

module rtl_pnl_bridge_tb;

    reg clk, rst_n;
    reg fill_valid, fill_side, mark_valid;
    reg [31:0] fill_symbol, fill_price, fill_volume;
    reg [31:0] mark_symbol, mark_price;
    reg [31:0] query_symbol;

    wire signed [31:0] query_net_position, query_unrealized_pnl, query_realized_pnl;
    wire signed [31:0] total_realized_pnl, total_unrealized_pnl;
    wire signed [31:0] net_position, daily_pnl;

    pnl_tracker #(.MAX_SYMBOLS(4)) dut (
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

    integer fv, fo;
    integer r_fill_valid, r_fill_side, r_fill_price, r_fill_volume, r_mark_price;

    initial begin
        clk = 0; rst_n = 0;
        fill_valid = 0; mark_valid = 0; fill_side = 0;
        fill_symbol = 32'h1; fill_price = 0; fill_volume = 0;
        mark_symbol = 32'h1; mark_price = 0;
        query_symbol = 32'h1;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        fv = $fopen("rtl_pnl_input.txt", "r");
        if (fv == 0) begin $display("ERROR: cannot open rtl_pnl_input.txt"); $finish; end
        r_fill_valid = 0; r_fill_side = 0; r_fill_price = 0; r_fill_volume = 0; r_mark_price = 0;
        $fscanf(fv, "%d,%d,%d,%d,%d\n",
            r_fill_valid, r_fill_side, r_fill_price, r_fill_volume, r_mark_price);
        $fclose(fv);

        fill_valid  = r_fill_valid[0];
        fill_side   = r_fill_side[0];
        fill_price  = r_fill_price;
        fill_volume = r_fill_volume;
        mark_valid  = 1;
        mark_price  = r_mark_price;

        @(posedge clk); #1;
        fill_valid = 0; mark_valid = 0;
        // extra cycles for unrealized scan
        repeat(4) @(posedge clk); #1;

        fo = $fopen("rtl_pnl_output.txt", "w");
        $fdisplay(fo, "%0d,%0d,%0d,%0d",
            total_realized_pnl, total_unrealized_pnl, net_position, daily_pnl);
        $fclose(fo);
        $finish;
    end
endmodule
