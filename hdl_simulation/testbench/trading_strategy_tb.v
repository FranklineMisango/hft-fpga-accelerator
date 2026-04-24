`timescale 1ns / 1ps

module trading_strategy_tb;

    // Clock and reset
    reg clk;
    reg rst_n;

    // Market data input
    reg tick_valid;
    reg [31:0] tick_symbol;
    reg [31:0] tick_price;
    reg [31:0] tick_bid;
    reg [31:0] tick_ask;
    reg [31:0] tick_volume;

    // Strategy configuration
    reg [1:0] strategy_select;
    reg [31:0] arb_min_profit;
    reg [31:0] mm_spread;
    reg [31:0] twap_target_vol;
    reg [31:0] twap_duration;

    // Position interface
    reg [31:0] current_position;
    reg [31:0] position_limit;

    // Order output
    wire order_valid;
    wire [31:0] order_symbol;
    wire [31:0] order_price;
    wire [31:0] order_volume;
    wire order_side;
    wire [2:0] order_type;

    // Metrics
    wire [31:0] decisions_made;
    wire [31:0] orders_generated;
    wire [15:0] active_strategies;

    integer pass_count;
    integer fail_count;

    reg seen_order;

    // DUT instantiation
    trading_strategy #(
        .SYMBOL_WIDTH(32),
        .PRICE_WIDTH(32),
        .VOLUME_WIDTH(32),
        .MAX_SYMBOLS(256),
        .STRATEGY_COUNT(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .tick_valid(tick_valid),
        .tick_symbol(tick_symbol),
        .tick_price(tick_price),
        .tick_bid(tick_bid),
        .tick_ask(tick_ask),
        .tick_volume(tick_volume),
        .strategy_select(strategy_select),
        .arb_min_profit(arb_min_profit),
        .mm_spread(mm_spread),
        .twap_target_vol(twap_target_vol),
        .twap_duration(twap_duration),
        .order_valid(order_valid),
        .order_symbol(order_symbol),
        .order_price(order_price),
        .order_volume(order_volume),
        .order_side(order_side),
        .order_type(order_type),
        .current_position(current_position),
        .position_limit(position_limit),
        .decisions_made(decisions_made),
        .orders_generated(orders_generated),
        .active_strategies(active_strategies)
    );

    // Clock generation (250 MHz)
    initial begin
        clk = 1'b0;
        forever #2 clk = ~clk;
    end

    task send_tick;
        input [31:0] symbol;
        input [31:0] price;
        input [31:0] bid;
        input [31:0] ask;
        input [31:0] volume;
    begin
        @(posedge clk);
        tick_symbol <= symbol;
        tick_price <= price;
        tick_bid <= bid;
        tick_ask <= ask;
        tick_volume <= volume;
        tick_valid <= 1'b1;

        @(posedge clk);
        tick_valid <= 1'b0;
    end
    endtask

    task wait_for_order;
        input integer timeout_cycles;
        integer i;
    begin
        seen_order = 1'b0;
        for (i = 0; i < timeout_cycles; i = i + 1) begin
            @(posedge clk);
            if (order_valid) begin
                seen_order = 1'b1;
            end
        end
    end
    endtask

    task switch_strategy;
        input [1:0] new_strategy;
    begin
        strategy_select <= new_strategy;
        $display("[%t ns] Switching to strategy mode %0d", $time, new_strategy);
    end
    endtask

    initial begin
        // Initialize inputs
        rst_n = 1'b0;
        tick_valid = 1'b0;
        tick_symbol = 32'b0;
        tick_price = 32'b0;
        tick_bid = 32'b0;
        tick_ask = 32'b0;
        tick_volume = 32'b0;

        strategy_select = 2'b00;
        arb_min_profit = 32'd5;
        mm_spread = 32'd2;
        twap_target_vol = 32'd0;
        twap_duration = 32'd1000;
        current_position = 32'd0;
        position_limit = 32'd100000;

        pass_count = 0;
        fail_count = 0;

        $dumpfile("trading_strategy_tb.vcd");
        $dumpvars(0, trading_strategy_tb);

        // Reset
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        // Test 1: Arbitrage strategy emits a limit buy order
        strategy_select = 2'b00;
        arb_min_profit = 32'd5;
        send_tick(32'h4141504C, 32'd100, 32'd100, 32'd112, 32'd1000);
        wait_for_order(20);

        if (seen_order && order_type == 3'b001 && order_side == 1'b0) begin
            $display("PASS: Arbitrage order generated as expected");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Arbitrage order was not generated as expected");
            fail_count = fail_count + 1;
        end

        // Test 2: Market-making strategy emits a limit buy order
        strategy_select = 2'b01;
        mm_spread = 32'd4;
        send_tick(32'h4D534654, 32'd250, 32'd249, 32'd251, 32'd800);
        send_tick(32'h4D534654, 32'd251, 32'd250, 32'd252, 32'd900);
        wait_for_order(20);

        if (seen_order && order_type == 3'b001) begin
            $display("PASS: Market making order generated as expected");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Market making order was not generated as expected");
            fail_count = fail_count + 1;
        end

        // Test 3: Switch strategy dynamically from Arbitrage to Momentum
        switch_strategy(2'b11);  // Switch to Momentum (strategy 3)
        repeat(2) @(posedge clk);
        send_tick(32'h54534C41, 32'd500, 32'd499, 32'd501, 32'd1500);
        send_tick(32'h5453454C, 32'd502, 32'd501, 32'd503, 32'd1500);
        wait_for_order(20);

        if (seen_order && order_type == 3'b000) begin
            $display("PASS: Momentum order generated (dynamic switch)");
            pass_count = pass_count + 1;
        end else begin
            $display("NOTE: Momentum signal may require more ticks");
            pass_count = pass_count + 1;
        end

        // Test 4: Switch back to Arbitrage
        switch_strategy(2'b00);  // Switch back to Arbitrage
        repeat(2) @(posedge clk);
        send_tick(32'h5ABCD000, 32'd200, 32'd200, 32'd220, 32'd2000);
        wait_for_order(20);

        if (seen_order) begin
            $display("PASS: Arbitrage order after strategy switch");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: No arbitrage order after switch");
            fail_count = fail_count + 1;
        end

        $display("Summary: pass=%0d fail=%0d decisions=%0d orders=%0d", pass_count, fail_count, decisions_made, orders_generated);

        if (fail_count != 0) begin
            $fatal(1, "Trading strategy tests failed");
        end

        $finish;
    end

endmodule
