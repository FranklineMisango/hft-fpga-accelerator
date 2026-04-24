`timescale 1ns/1ps
// Bridge harness for risk_manager.
// Reads one line from rtl_risk_input.txt:
//   <order_valid>,<order_side>,<order_volume>,<order_price>,
//   <net_position>,<daily_pnl>,<cfg_max_position>,<cfg_max_order_size>,
//   <cfg_max_drawdown>,<cfg_kill_switch>
// Writes one line to rtl_risk_output.txt:
//   <order_approved>,<reject_reason>,<kill_active>,<approved_cnt>,<rejected_cnt>

module rtl_risk_bridge_tb;

    reg clk, rst_n;
    reg order_valid, order_side, cfg_kill_switch;
    reg [31:0] order_volume, order_price;
    reg signed [31:0] net_position, unrealized_pnl, daily_pnl;
    reg [31:0] cfg_max_position, cfg_max_order_size, cfg_max_drawdown;
    reg [15:0] cfg_max_order_rate;

    wire order_approved, kill_active;
    wire [4:0] reject_reason;
    wire [31:0] orders_approved_cnt, orders_rejected_cnt;

    risk_manager dut (
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
        .order_approved(order_approved),
        .reject_reason(reject_reason),
        .kill_active(kill_active),
        .orders_approved_cnt(orders_approved_cnt),
        .orders_rejected_cnt(orders_rejected_cnt)
    );

    always #5 clk = ~clk;

    integer fv, fo;
    integer r_valid, r_side, r_vol, r_price, r_pos, r_dpnl;
    integer r_max_pos, r_max_sz, r_max_dd, r_kill;

    initial begin
        clk = 0; rst_n = 0;
        order_valid = 0; order_side = 0; cfg_kill_switch = 0;
        order_volume = 0; order_price = 0;
        net_position = 0; unrealized_pnl = 0; daily_pnl = 0;
        cfg_max_position = 32'd100000; cfg_max_order_size = 32'd10000;
        cfg_max_drawdown = 32'd50000; cfg_max_order_rate = 16'd1000;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        fv = $fopen("rtl_risk_input.txt", "r");
        if (fv == 0) begin $display("ERROR: cannot open rtl_risk_input.txt"); $finish; end
        r_valid=0; r_side=0; r_vol=0; r_price=0; r_pos=0; r_dpnl=0;
        r_max_pos=0; r_max_sz=0; r_max_dd=0; r_kill=0;
        $fscanf(fv, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
            r_valid, r_side, r_vol, r_price, r_pos, r_dpnl,
            r_max_pos, r_max_sz, r_max_dd, r_kill);
        $fclose(fv);

        order_valid       = r_valid[0];
        order_side        = r_side[0];
        order_volume      = r_vol;
        order_price       = r_price;
        net_position      = r_pos;
        daily_pnl         = r_dpnl;
        cfg_max_position  = r_max_pos;
        cfg_max_order_size= r_max_sz;
        cfg_max_drawdown  = r_max_dd;
        cfg_kill_switch   = r_kill[0];

        @(posedge clk); #1;
        order_valid = 0;
        repeat(3) @(posedge clk); #1;

        fo = $fopen("rtl_risk_output.txt", "w");
        $fdisplay(fo, "%0d,%0d,%0d,%0d,%0d",
            order_approved, reject_reason, kill_active,
            orders_approved_cnt, orders_rejected_cnt);
        $fclose(fo);
        $finish;
    end
endmodule
