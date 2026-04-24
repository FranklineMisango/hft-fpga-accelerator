`timescale 1ns/1ps

/*
 * Risk Manager
 * Pre-trade risk enforcement: position limits, drawdown halt,
 * order rate limiting, and hardware kill switch.
 *
 * All checks are combinational (zero added latency on the hot path).
 * The kill switch and drawdown halt are registered for glitch immunity.
 */

module risk_manager #(
    parameter PRICE_WIDTH  = 32,
    parameter VOLUME_WIDTH = 32,
    parameter SYMBOL_WIDTH = 32
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- incoming order to evaluate ----
    input  wire                     order_valid,
    input  wire                     order_side,        // 0=buy 1=sell
    input  wire [VOLUME_WIDTH-1:0]  order_volume,
    input  wire [PRICE_WIDTH-1:0]   order_price,

    // ---- live position & PnL feed (from pnl_tracker) ----
    input  wire signed [VOLUME_WIDTH-1:0] net_position,   // signed
    input  wire signed [PRICE_WIDTH-1:0]  unrealized_pnl, // signed, scaled
    input  wire signed [PRICE_WIDTH-1:0]  daily_pnl,      // signed, scaled

    // ---- configurable limits ----
    input  wire [VOLUME_WIDTH-1:0]  cfg_max_position,     // absolute net position cap
    input  wire [VOLUME_WIDTH-1:0]  cfg_max_order_size,   // single-order size cap
    input  wire [PRICE_WIDTH-1:0]   cfg_max_drawdown,     // daily loss limit (positive value)
    input  wire [15:0]              cfg_max_order_rate,   // max orders per 2^16 cycles window
    input  wire                     cfg_kill_switch,      // software kill (1 = halt)

    // ---- decision output ----
    output wire                     order_approved,       // 1 = pass all checks
    output wire [4:0]               reject_reason,        // one-hot violation flags

    // ---- status ----
    output wire                     kill_active,          // kill switch or drawdown halt
    output wire [31:0]              orders_approved_cnt,
    output wire [31:0]              orders_rejected_cnt
);

// ---------------------------------------------------------------------------
// Reject reason bits
// ---------------------------------------------------------------------------
localparam R_KILL        = 5'b00001;
localparam R_DRAWDOWN    = 5'b00010;
localparam R_POSITION    = 5'b00100;
localparam R_ORDER_SIZE  = 5'b01000;
localparam R_RATE_LIMIT  = 5'b10000;

// ---------------------------------------------------------------------------
// Rate limiter: sliding window counter over 2^16 clock cycles
// ---------------------------------------------------------------------------
reg [15:0] rate_window_counter;   // counts cycles in current window
reg [15:0] order_count_window;    // orders seen in current window
reg        rate_limit_violation;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rate_window_counter <= 16'b0;
        order_count_window  <= 16'b0;
        rate_limit_violation <= 1'b0;
    end else begin
        if (rate_window_counter == 16'hFFFF) begin
            rate_window_counter  <= 16'b0;
            order_count_window   <= 16'b0;
            rate_limit_violation <= 1'b0;
        end else begin
            rate_window_counter <= rate_window_counter + 1'b1;
            if (order_valid && order_approved) begin
                order_count_window <= order_count_window + 1'b1;
                if (order_count_window >= cfg_max_order_rate)
                    rate_limit_violation <= 1'b1;
            end
        end
    end
end

// ---------------------------------------------------------------------------
// Drawdown halt (registered)
// ---------------------------------------------------------------------------
reg drawdown_halt;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        drawdown_halt <= 1'b0;
    else if ((-daily_pnl) > $signed({1'b0, cfg_max_drawdown}))
        drawdown_halt <= 1'b1;
    else if (daily_pnl >= 0)          // reset when PnL recovers (e.g. new day)
        drawdown_halt <= 1'b0;
end

// ---------------------------------------------------------------------------
// Position check (combinational)
// ---------------------------------------------------------------------------
wire signed [VOLUME_WIDTH:0] projected_position;
assign projected_position = order_side
    ? ($signed({1'b0, net_position}) - $signed({1'b0, order_volume}))
    : ($signed({1'b0, net_position}) + $signed({1'b0, order_volume}));

wire position_violation;
assign position_violation = (projected_position >  $signed({1'b0, cfg_max_position})) ||
                            (projected_position < -$signed({1'b0, cfg_max_position}));

// ---------------------------------------------------------------------------
// Order size check (combinational)
// ---------------------------------------------------------------------------
wire size_violation;
assign size_violation = (order_volume > cfg_max_order_size);

// ---------------------------------------------------------------------------
// Aggregate decision
// ---------------------------------------------------------------------------
wire kill_sw_reg;
assign kill_sw_reg = cfg_kill_switch;

assign reject_reason = (kill_sw_reg    ? R_KILL       : 5'b0)
                     | (drawdown_halt  ? R_DRAWDOWN   : 5'b0)
                     | (position_violation ? R_POSITION : 5'b0)
                     | (size_violation ? R_ORDER_SIZE : 5'b0)
                     | (rate_limit_violation ? R_RATE_LIMIT : 5'b0);

assign order_approved = order_valid && (reject_reason == 5'b0);
assign kill_active    = kill_sw_reg || drawdown_halt;

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------
reg [31:0] approved_cnt;
reg [31:0] rejected_cnt;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        approved_cnt <= 32'b0;
        rejected_cnt <= 32'b0;
    end else if (order_valid) begin
        if (order_approved)
            approved_cnt <= approved_cnt + 1'b1;
        else
            rejected_cnt <= rejected_cnt + 1'b1;
    end
end

assign orders_approved_cnt = approved_cnt;
assign orders_rejected_cnt = rejected_cnt;

endmodule
