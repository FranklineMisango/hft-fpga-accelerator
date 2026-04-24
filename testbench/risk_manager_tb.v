`timescale 1ns/1ps

module risk_manager_tb;

    // Parameters
    localparam PRICE_WIDTH  = 32;
    localparam VOLUME_WIDTH = 32;
    localparam SYMBOL_WIDTH = 32;

    reg clk, rst_n;
    reg order_valid, order_side;
    reg [VOLUME_WIDTH-1:0] order_volume;
    reg [PRICE_WIDTH-1:0]  order_price;
    reg signed [VOLUME_WIDTH-1:0] net_position;
    reg signed [PRICE_WIDTH-1:0]  unrealized_pnl, daily_pnl;
    reg [VOLUME_WIDTH-1:0] cfg_max_position, cfg_max_order_size;
    reg [PRICE_WIDTH-1:0]  cfg_max_drawdown;
    reg [15:0]             cfg_max_order_rate;
    reg                    cfg_kill_switch;

    wire                   order_approved;
    wire [4:0]             reject_reason;
    wire                   kill_active;
    wire [31:0]            orders_approved_cnt, orders_rejected_cnt;

    risk_manager #(
        .PRICE_WIDTH(PRICE_WIDTH),
        .VOLUME_WIDTH(VOLUME_WIDTH),
        .SYMBOL_WIDTH(SYMBOL_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .order_valid(order_valid), .order_side(order_side),
        .order_volume(order_volume), .order_price(order_price),
        .net_position(net_position),
        .unrealized_pnl(unrealized_pnl), .daily_pnl(daily_pnl),
        .cfg_max_position(cfg_max_position),
        .cfg_max_order_size(cfg_max_order_size),
        .cfg_max_drawdown(cfg_max_drawdown),
        .cfg_max_order_rate(cfg_max_order_rate),
        .cfg_kill_switch(cfg_kill_switch),
        .order_approved(order_approved), .reject_reason(reject_reason),
        .kill_active(kill_active),
        .orders_approved_cnt(orders_approved_cnt),
        .orders_rejected_cnt(orders_rejected_cnt)
    );

    always #5 clk = ~clk;

    task send_order;
        input side;
        input [VOLUME_WIDTH-1:0] vol;
        input [PRICE_WIDTH-1:0]  price;
        begin
            order_valid  = 1'b1;
            order_side   = side;
            order_volume = vol;
            order_price  = price;
            @(posedge clk); #1;
            order_valid  = 1'b0;
        end
    endtask

    integer pass_count, fail_count;

    initial begin
        clk = 0; rst_n = 0; pass_count = 0; fail_count = 0;
        order_valid = 0; order_side = 0;
        order_volume = 0; order_price = 0;
        net_position = 0; unrealized_pnl = 0; daily_pnl = 0;
        cfg_max_position  = 32'd10000;
        cfg_max_order_size= 32'd1000;
        cfg_max_drawdown  = 32'd5000;
        cfg_max_order_rate= 16'd100;
        cfg_kill_switch   = 1'b0;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        // Test 1: Normal order should pass
        send_order(1'b0, 32'd500, 32'd100);
        if (order_approved) begin
            $display("PASS: Normal order approved"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Normal order rejected (reason=%b)", reject_reason); fail_count = fail_count + 1;
        end

        // Test 2: Oversized order should be rejected
        send_order(1'b0, 32'd2000, 32'd100);
        if (!order_approved && reject_reason[3]) begin
            $display("PASS: Oversized order rejected"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Oversized order not caught (approved=%b reason=%b)", order_approved, reject_reason);
            fail_count = fail_count + 1;
        end

        // Test 3: Kill switch
        cfg_kill_switch = 1'b1;
        send_order(1'b0, 32'd100, 32'd100);
        if (!order_approved && reject_reason[0]) begin
            $display("PASS: Kill switch blocks order"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Kill switch not effective"); fail_count = fail_count + 1;
        end
        cfg_kill_switch = 1'b0;

        // Test 4: Drawdown halt
        daily_pnl = -32'sd6000;
        @(posedge clk); @(posedge clk); #1;
        send_order(1'b0, 32'd100, 32'd100);
        if (!order_approved) begin
            $display("PASS: Drawdown halt blocks order"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Drawdown halt not triggered"); fail_count = fail_count + 1;
        end
        daily_pnl = 32'sd0;
        @(posedge clk); @(posedge clk); #1;

        // Test 5: Position limit
        net_position = 32'sd9800;
        send_order(1'b0, 32'd500, 32'd100); // would push to 10300 > 10000
        if (!order_approved && reject_reason[2]) begin
            $display("PASS: Position limit enforced"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Position limit not enforced (approved=%b reason=%b)", order_approved, reject_reason);
            fail_count = fail_count + 1;
        end
        net_position = 32'sd0;

        $display("---");
        $display("risk_manager_tb: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("PASS: All risk_manager tests passed");
        else                 $display("FAIL: Some risk_manager tests failed");
        $finish;
    end

endmodule
