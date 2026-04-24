`timescale 1ns/1ps

/*
 * PnL Tracker
 * Tracks realized and unrealized PnL, net position, and VWAP cost basis
 * per symbol using a fixed-size position table.
 *
 * Realized PnL is updated on every fill.
 * Unrealized PnL is recomputed each cycle from the latest mid-price feed.
 */

module pnl_tracker #(
    parameter PRICE_WIDTH   = 32,
    parameter VOLUME_WIDTH  = 32,
    parameter SYMBOL_WIDTH  = 32,
    parameter MAX_SYMBOLS   = 16,   // number of tracked symbols
    parameter PRICE_SCALE   = 100   // price units per dollar (e.g. cents)
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- fill feed (from order_manager) ----
    input  wire                     fill_valid,
    input  wire [SYMBOL_WIDTH-1:0]  fill_symbol,
    input  wire [PRICE_WIDTH-1:0]   fill_price,
    input  wire [VOLUME_WIDTH-1:0]  fill_volume,
    input  wire                     fill_side,         // 0=buy 1=sell

    // ---- mark-to-market price feed ----
    input  wire                     mark_valid,
    input  wire [SYMBOL_WIDTH-1:0]  mark_symbol,
    input  wire [PRICE_WIDTH-1:0]   mark_price,

    // ---- query interface (combinational read) ----
    input  wire [SYMBOL_WIDTH-1:0]  query_symbol,
    output wire signed [VOLUME_WIDTH-1:0]  query_net_position,
    output wire signed [PRICE_WIDTH-1:0]   query_unrealized_pnl,
    output wire signed [PRICE_WIDTH-1:0]   query_realized_pnl,

    // ---- aggregate outputs ----
    output reg  signed [PRICE_WIDTH-1:0]   total_realized_pnl,
    output reg  signed [PRICE_WIDTH-1:0]   total_unrealized_pnl,
    output reg  signed [VOLUME_WIDTH-1:0]  net_position,      // aggregate signed position
    output reg  signed [PRICE_WIDTH-1:0]   daily_pnl          // realized + unrealized
);

// ---------------------------------------------------------------------------
// Symbol table
// ---------------------------------------------------------------------------
reg [SYMBOL_WIDTH-1:0]          sym_id       [0:MAX_SYMBOLS-1];
reg signed [VOLUME_WIDTH-1:0]   sym_position [0:MAX_SYMBOLS-1]; // signed net qty
reg [PRICE_WIDTH-1:0]           sym_vwap     [0:MAX_SYMBOLS-1]; // avg cost basis
reg signed [PRICE_WIDTH-1:0]    sym_realized [0:MAX_SYMBOLS-1];
reg signed [PRICE_WIDTH-1:0]    sym_mark     [0:MAX_SYMBOLS-1]; // last mark price
reg                             sym_valid    [0:MAX_SYMBOLS-1];
reg [$clog2(MAX_SYMBOLS)-1:0]   sym_count;

integer i;

// ---------------------------------------------------------------------------
// Symbol lookup (combinational)
// ---------------------------------------------------------------------------
reg [$clog2(MAX_SYMBOLS)-1:0] fill_idx;
reg                           fill_found;
reg [$clog2(MAX_SYMBOLS)-1:0] mark_idx;
reg                           mark_found;
reg [$clog2(MAX_SYMBOLS)-1:0] query_idx;
reg                           query_found;

always @(*) begin : lookup_fill
    fill_found = 1'b0;
    fill_idx   = {$clog2(MAX_SYMBOLS){1'b0}};
    for (i = 0; i < MAX_SYMBOLS; i = i + 1)
        if (sym_valid[i] && sym_id[i] == fill_symbol) begin
            fill_found = 1'b1;
            fill_idx   = i[$clog2(MAX_SYMBOLS)-1:0];
        end
end

always @(*) begin : lookup_mark
    mark_found = 1'b0;
    mark_idx   = {$clog2(MAX_SYMBOLS){1'b0}};
    for (i = 0; i < MAX_SYMBOLS; i = i + 1)
        if (sym_valid[i] && sym_id[i] == mark_symbol) begin
            mark_found = 1'b1;
            mark_idx   = i[$clog2(MAX_SYMBOLS)-1:0];
        end
end

always @(*) begin : lookup_query
    query_found = 1'b0;
    query_idx   = {$clog2(MAX_SYMBOLS){1'b0}};
    for (i = 0; i < MAX_SYMBOLS; i = i + 1)
        if (sym_valid[i] && sym_id[i] == query_symbol) begin
            query_found = 1'b1;
            query_idx   = i[$clog2(MAX_SYMBOLS)-1:0];
        end
end

// ---------------------------------------------------------------------------
// Query outputs (combinational)
// ---------------------------------------------------------------------------
assign query_net_position  = query_found ? sym_position[query_idx] : {VOLUME_WIDTH{1'b0}};
assign query_realized_pnl  = query_found ? sym_realized[query_idx] : {PRICE_WIDTH{1'b0}};
assign query_unrealized_pnl = query_found
    ? ($signed(sym_mark[query_idx]) - $signed(sym_vwap[query_idx]))
      * sym_position[query_idx]
    : {PRICE_WIDTH{1'b0}};

// ---------------------------------------------------------------------------
// Fill processing
// ---------------------------------------------------------------------------
reg signed [PRICE_WIDTH-1:0]  realized_delta;
reg signed [VOLUME_WIDTH-1:0] new_position;
reg [PRICE_WIDTH-1:0]         new_vwap;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sym_count <= {$clog2(MAX_SYMBOLS){1'b0}};
        total_realized_pnl   <= {PRICE_WIDTH{1'b0}};
        total_unrealized_pnl <= {PRICE_WIDTH{1'b0}};
        net_position         <= {VOLUME_WIDTH{1'b0}};
        daily_pnl            <= {PRICE_WIDTH{1'b0}};
        for (i = 0; i < MAX_SYMBOLS; i = i + 1) begin
            sym_valid[i]    <= 1'b0;
            sym_position[i] <= {VOLUME_WIDTH{1'b0}};
            sym_vwap[i]     <= {PRICE_WIDTH{1'b0}};
            sym_realized[i] <= {PRICE_WIDTH{1'b0}};
            sym_mark[i]     <= {PRICE_WIDTH{1'b0}};
            sym_id[i]       <= {SYMBOL_WIDTH{1'b0}};
        end
    end else begin

        // ---- fill update ----
        if (fill_valid) begin
            if (fill_found) begin
                new_position = fill_side
                    ? (sym_position[fill_idx] - $signed({1'b0, fill_volume}))
                    : (sym_position[fill_idx] + $signed({1'b0, fill_volume}));

                // Realized PnL on reducing trades
                if (fill_side && sym_position[fill_idx] > 0) begin
                    // selling into long: realize (fill_price - vwap) * min(fill_vol, long_pos)
                    realized_delta = ($signed({1'b0, fill_price}) - $signed({1'b0, sym_vwap[fill_idx]}))
                                     * $signed({1'b0, fill_volume});
                    sym_realized[fill_idx] <= sym_realized[fill_idx] + realized_delta;
                    total_realized_pnl     <= total_realized_pnl     + realized_delta;
                end else if (!fill_side && sym_position[fill_idx] < 0) begin
                    // buying into short: realize (vwap - fill_price) * fill_vol
                    realized_delta = ($signed({1'b0, sym_vwap[fill_idx]}) - $signed({1'b0, fill_price}))
                                     * $signed({1'b0, fill_volume});
                    sym_realized[fill_idx] <= sym_realized[fill_idx] + realized_delta;
                    total_realized_pnl     <= total_realized_pnl     + realized_delta;
                end

                // Update VWAP on position-increasing trades
                if (!fill_side) begin
                    if (sym_position[fill_idx] >= 0)
                        new_vwap = (sym_vwap[fill_idx] * sym_position[fill_idx][VOLUME_WIDTH-1:0]
                                    + fill_price * fill_volume)
                                   / (sym_position[fill_idx][VOLUME_WIDTH-1:0] + fill_volume);
                    else
                        new_vwap = fill_price;
                    sym_vwap[fill_idx] <= new_vwap;
                end

                sym_position[fill_idx] <= new_position;

            end else if (sym_count < MAX_SYMBOLS) begin
                // New symbol
                sym_id[sym_count]       <= fill_symbol;
                sym_valid[sym_count]    <= 1'b1;
                sym_position[sym_count] <= fill_side
                    ? -$signed({1'b0, fill_volume})
                    :  $signed({1'b0, fill_volume});
                sym_vwap[sym_count]     <= fill_price;
                sym_realized[sym_count] <= {PRICE_WIDTH{1'b0}};
                sym_mark[sym_count]     <= fill_price;
                sym_count               <= sym_count + 1'b1;
            end
        end

        // ---- mark-to-market update ----
        if (mark_valid && mark_found)
            sym_mark[mark_idx] <= mark_price;

        // ---- aggregate unrealized PnL (full table scan each cycle) ----
        begin : unrealized_scan
            reg signed [PRICE_WIDTH-1:0] upnl_acc;
            reg signed [VOLUME_WIDTH-1:0] pos_acc;
            upnl_acc = {PRICE_WIDTH{1'b0}};
            pos_acc  = {VOLUME_WIDTH{1'b0}};
            for (i = 0; i < MAX_SYMBOLS; i = i + 1) begin
                if (sym_valid[i]) begin
                    upnl_acc = upnl_acc
                        + ($signed({1'b0, sym_mark[i]}) - $signed({1'b0, sym_vwap[i]}))
                          * sym_position[i];
                    pos_acc = pos_acc + sym_position[i];
                end
            end
            total_unrealized_pnl <= upnl_acc;
            net_position         <= pos_acc;
            daily_pnl            <= total_realized_pnl + upnl_acc;
        end

    end
end

endmodule
