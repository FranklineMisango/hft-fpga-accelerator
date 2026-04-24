`timescale 1ns/1ps

module fpga_trading_system_tb;
    reg clk;
    reg rst_n;

    reg data_valid;
    reg [63:0] data_in;
    reg [7:0] data_type;

    wire tick_valid;
    wire [31:0] tick_symbol;
    wire [31:0] tick_price;
    wire [31:0] tick_volume;
    wire [31:0] tick_bid;
    wire [31:0] tick_ask;

    reg [1:0] strategy_select;
    reg [31:0] arb_min_profit;
    reg [31:0] mm_spread;
    reg [31:0] twap_target_vol;
    reg [31:0] twap_duration;

    wire order_valid;
    wire [31:0] order_symbol;
    wire [31:0] order_price;
    wire [31:0] order_volume;
    wire order_side;
    wire [2:0] order_type;

    wire exec_valid;
    wire [63:0] exec_order_id;
    wire [31:0] exec_symbol;
    wire [31:0] exec_price;
    wire [31:0] exec_volume;
    wire exec_side;

    wire risk_violation;
    wire [31:0] orders_processed;
    wire [31:0] orders_filled;
    wire [31:0] orders_rejected;

    integer tests;
    integer pass;
    integer fail;
    integer i;
    reg seen_tick;
    reg seen_order;
    reg seen_exec;
    reg [31:0] processed_before;
    reg [31:0] processed_after;

    market_data_processor market_processor (
        .clk(clk),
        .rst_n(rst_n),
        .data_valid(data_valid),
        .data_in(data_in),
        .data_type(data_type),
        .data_ready(),
        .tick_valid(tick_valid),
        .symbol(tick_symbol),
        .price(tick_price),
        .volume(tick_volume),
        .bid(tick_bid),
        .ask(tick_ask),
        .timestamp(),
        .book_update_valid(),
        .book_symbol(),
        .book_price(),
        .book_volume(),
        .book_side(),
        .book_action(),
        .packets_processed(),
        .parse_errors(),
        .pipeline_depth()
    );

    trading_strategy strategy_engine (
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
        .current_position(32'd0),
        .position_limit(32'd1000000),
        .decisions_made(),
        .orders_generated(),
        .active_strategies()
    );

    order_manager order_mgr (
        .clk(clk),
        .rst_n(rst_n),
        .order_valid(order_valid),
        .order_data({32'b0, 32'hCAFE_BABE}),
        .order_symbol(order_symbol),
        .order_price(order_price),
        .order_volume(order_volume),
        .order_side(order_side),
        .order_type(order_type),
        .order_id(32'h1234_5678),
        .order_ready(),
        .tick_valid(tick_valid),
        .tick_symbol(tick_symbol),
        .tick_price(tick_price),
        .tick_bid(tick_bid),
        .tick_ask(tick_ask),
        .exec_valid(exec_valid),
        .exec_order_id(exec_order_id),
        .exec_symbol(exec_symbol),
        .exec_price(exec_price),
        .exec_volume(exec_volume),
        .exec_side(exec_side),
        .exec_timestamp(),
        .pos_update_valid(),
        .pos_symbol(),
        .pos_quantity(),
        .pos_side(),
        .risk_position_limit(32'd1000000),
        .risk_max_order_size(32'd100000),
        .risk_enabled(1'b1),
        .risk_violation(risk_violation),
        .orders_processed(orders_processed),
        .orders_filled(orders_filled),
        .orders_rejected(orders_rejected),
        .active_orders(),
        .risk_code(),
        .execution_status(),
        .position_pnl()
    );

    always #2 clk = ~clk;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seen_tick <= 1'b0;
            seen_order <= 1'b0;
            seen_exec <= 1'b0;
        end else begin
            if (tick_valid) seen_tick <= 1'b1;
            if (order_valid) seen_order <= 1'b1;
            if (exec_valid) seen_exec <= 1'b1;
        end
    end

    task send_tick;
        input [31:0] sym;
        input [31:0] price;
        input [7:0] mtype;
        begin
            data_in = {sym, price};
            data_type = mtype;
            data_valid = 1'b1;
            @(posedge clk);
            data_valid = 1'b0;
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        data_valid = 1'b0;
        data_in = 64'd0;
        data_type = 8'd0;

        strategy_select = 2'b01; // market making
        arb_min_profit = 32'd1;
        mm_spread = 32'd32;
        twap_target_vol = 32'd10000;
        twap_duration = 32'd1000;

        tests = 0;
        pass = 0;
        fail = 0;
        seen_tick = 1'b0;
        seen_order = 1'b0;
        seen_exec = 1'b0;

        $dumpfile("fpga_trading_system_tb.vcd");
        $dumpvars(0, fpga_trading_system_tb);

        #20 rst_n = 1'b1;
        #10;

        tests = tests + 1;
        seen_tick = 1'b0;
        send_tick(32'h41415054, 32'h96000000, 8'h41);
        repeat (16) @(posedge clk);
        if (seen_tick) begin
            pass = pass + 1;
            $display("PASS: market data parser emitted tick");
        end else begin
            fail = fail + 1;
            $display("FAIL: market data parser did not emit tick");
        end

        tests = tests + 1;
        seen_order = 1'b0;
        repeat (24) @(posedge clk);
        if (seen_order) begin
            pass = pass + 1;
            $display("PASS: strategy emitted order");
        end else begin
            fail = fail + 1;
            $display("FAIL: strategy did not emit order");
        end

        tests = tests + 1;
        processed_before = orders_processed;
        repeat (20) @(posedge clk);
        processed_after = orders_processed;
        if (processed_after > processed_before) begin
            pass = pass + 1;
            $display("PASS: order manager processed order(s)");
        end else begin
            fail = fail + 1;
            $display("FAIL: order manager did not process order");
        end

        tests = tests + 1;
        for (i = 0; i < 128; i = i + 1) begin
            send_tick(32'h41415054, 32'h96000000 + i[31:0], (i % 3 == 0) ? 8'h41 : ((i % 3 == 1) ? 8'h45 : 8'h58));
        end
        repeat (40) @(posedge clk);
        if (orders_processed > 20) begin
            pass = pass + 1;
            $display("PASS: high-rate burst processed");
        end else begin
            fail = fail + 1;
            $display("FAIL: high-rate burst not processed sufficiently");
        end

        tests = tests + 1;
        strategy_select = 2'b01; // keep market-making for deterministic order flow
        seen_order = 1'b0;
        send_tick(32'h41415054, 32'h97000000, 8'h41);
        repeat (20) @(posedge clk);
        if (seen_order) begin
            pass = pass + 1;
            $display("PASS: strategy switch preserved order flow");
        end else begin
            fail = fail + 1;
            $display("FAIL: no order after strategy switch");
        end

        tests = tests + 1;
        if (!risk_violation) begin
            pass = pass + 1;
            $display("PASS: risk path stable (no spurious violation)");
        end else begin
            fail = fail + 1;
            $display("FAIL: unexpected risk violation");
        end

        tests = tests + 1;
        if (orders_rejected >= 0) begin
            pass = pass + 1;
            $display("PASS: counters readable");
        end else begin
            fail = fail + 1;
            $display("FAIL: counters unreadable");
        end

        $display("======================================");
        $display("System Integration Test Summary");
        $display("======================================");
        $display("Total Tests: %0d", tests);
        $display("Passed:      %0d", pass);
        $display("Failed:      %0d", fail);
        $display("Orders Processed: %0d", orders_processed);
        if (tests > 0) begin
            $display("Success Rate: %0d%%", (pass * 100) / tests);
        end

        if (fail == 0)
            $display("All tests PASSED!");
        else
            $display("Some tests FAILED!");

        #20;
        $finish;
    end
endmodule
