`timescale 1ns/1ps

/*
 * Order Book
 * Price-level aggregated order book with best bid/ask maintenance.
 * Supports add, modify, and delete actions from market_data_processor.
 *
 * Implemented as two sorted arrays (bids descending, asks ascending)
 * with a configurable number of price levels.
 * Best bid/ask are maintained as registered outputs for zero-latency reads.
 */

module order_book #(
    parameter PRICE_WIDTH  = 32,
    parameter VOLUME_WIDTH = 32,
    parameter SYMBOL_WIDTH = 32,
    parameter DEPTH        = 16    // number of price levels per side
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- update feed (from market_data_processor) ----
    input  wire                     update_valid,
    input  wire [SYMBOL_WIDTH-1:0]  update_symbol,
    input  wire [PRICE_WIDTH-1:0]   update_price,
    input  wire [VOLUME_WIDTH-1:0]  update_volume,
    input  wire                     update_side,    // 0=bid 1=ask
    input  wire [1:0]               update_action,  // 0=add/modify 1=delete 2=clear

    // ---- best bid/ask outputs (registered, 1-cycle latency) ----
    output reg  [PRICE_WIDTH-1:0]   best_bid,
    output reg  [VOLUME_WIDTH-1:0]  best_bid_qty,
    output reg  [PRICE_WIDTH-1:0]   best_ask,
    output reg  [VOLUME_WIDTH-1:0]  best_ask_qty,
    output reg                      book_valid,     // 1 when both sides have at least one level

    // ---- mid price (combinational) ----
    output wire [PRICE_WIDTH-1:0]   mid_price,
    output wire [PRICE_WIDTH-1:0]   spread,

    // ---- depth query (combinational, level 0 = best) ----
    input  wire [$clog2(DEPTH)-1:0] query_level,
    input  wire                     query_side,     // 0=bid 1=ask
    output wire [PRICE_WIDTH-1:0]   query_price,
    output wire [VOLUME_WIDTH-1:0]  query_volume,

    // ---- statistics ----
    output wire [31:0]              update_count,
    output wire [15:0]              bid_levels,
    output wire [15:0]              ask_levels
);

// ---------------------------------------------------------------------------
// Price level storage
// Bids: sorted descending (index 0 = best bid)
// Asks: sorted ascending  (index 0 = best ask)
// ---------------------------------------------------------------------------
reg [PRICE_WIDTH-1:0]  bid_price  [0:DEPTH-1];
reg [VOLUME_WIDTH-1:0] bid_volume [0:DEPTH-1];
reg                    bid_valid  [0:DEPTH-1];

reg [PRICE_WIDTH-1:0]  ask_price  [0:DEPTH-1];
reg [VOLUME_WIDTH-1:0] ask_volume [0:DEPTH-1];
reg                    ask_valid  [0:DEPTH-1];

reg [$clog2(DEPTH):0]  bid_count;
reg [$clog2(DEPTH):0]  ask_count;
reg [31:0]             upd_counter;

integer i, j;

// ---------------------------------------------------------------------------
// Depth query (combinational)
// ---------------------------------------------------------------------------
assign query_price  = query_side
    ? (ask_valid[query_level] ? ask_price[query_level]  : {PRICE_WIDTH{1'b0}})
    : (bid_valid[query_level] ? bid_price[query_level]  : {PRICE_WIDTH{1'b0}});
assign query_volume = query_side
    ? (ask_valid[query_level] ? ask_volume[query_level] : {VOLUME_WIDTH{1'b0}})
    : (bid_valid[query_level] ? bid_volume[query_level] : {VOLUME_WIDTH{1'b0}});

// ---------------------------------------------------------------------------
// Mid / spread (combinational)
// ---------------------------------------------------------------------------
assign mid_price = (book_valid) ? ((best_bid + best_ask) >> 1) : {PRICE_WIDTH{1'b0}};
assign spread    = (book_valid && best_ask >= best_bid) ? (best_ask - best_bid) : {PRICE_WIDTH{1'b0}};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
assign update_count = upd_counter;
assign bid_levels   = {{(16-$clog2(DEPTH)-1){1'b0}}, bid_count};
assign ask_levels   = {{(16-$clog2(DEPTH)-1){1'b0}}, ask_count};

// ---------------------------------------------------------------------------
// Update logic
// ---------------------------------------------------------------------------
// Action encodings
localparam ACT_UPSERT = 2'b00;
localparam ACT_DELETE = 2'b01;
localparam ACT_CLEAR  = 2'b10;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        bid_count   <= {($clog2(DEPTH)+1){1'b0}};
        ask_count   <= {($clog2(DEPTH)+1){1'b0}};
        upd_counter <= 32'b0;
        best_bid    <= {PRICE_WIDTH{1'b0}};
        best_bid_qty<= {VOLUME_WIDTH{1'b0}};
        best_ask    <= {PRICE_WIDTH{1'b1}};   // max value sentinel
        best_ask_qty<= {VOLUME_WIDTH{1'b0}};
        book_valid  <= 1'b0;
        for (i = 0; i < DEPTH; i = i + 1) begin
            bid_valid[i]  <= 1'b0;
            ask_valid[i]  <= 1'b0;
            bid_price[i]  <= {PRICE_WIDTH{1'b0}};
            bid_volume[i] <= {VOLUME_WIDTH{1'b0}};
            ask_price[i]  <= {PRICE_WIDTH{1'b0}};
            ask_volume[i] <= {VOLUME_WIDTH{1'b0}};
        end
    end else if (update_valid) begin
        upd_counter <= upd_counter + 1'b1;

        if (update_action == ACT_CLEAR) begin
            // Clear entire side
            if (!update_side) begin
                bid_count <= {($clog2(DEPTH)+1){1'b0}};
                for (i = 0; i < DEPTH; i = i + 1) bid_valid[i] <= 1'b0;
            end else begin
                ask_count <= {($clog2(DEPTH)+1){1'b0}};
                for (i = 0; i < DEPTH; i = i + 1) ask_valid[i] <= 1'b0;
            end

        end else if (!update_side) begin
            // ---- BID side ----
            if (update_action == ACT_DELETE) begin
                // Remove matching price level
                for (i = 0; i < DEPTH; i = i + 1) begin
                    if (bid_valid[i] && bid_price[i] == update_price) begin
                        bid_valid[i]  <= 1'b0;
                        bid_price[i]  <= {PRICE_WIDTH{1'b0}};
                        bid_volume[i] <= {VOLUME_WIDTH{1'b0}};
                        if (bid_count > 0) bid_count <= bid_count - 1'b1;
                    end
                end
            end else begin
                // Upsert: find existing level or insert sorted (descending)
                begin : bid_upsert
                    reg found;
                    reg [$clog2(DEPTH)-1:0] ins_pos;
                    found   = 1'b0;
                    ins_pos = {$clog2(DEPTH){1'b0}};

                    // Check for existing level
                    for (i = 0; i < DEPTH; i = i + 1) begin
                        if (bid_valid[i] && bid_price[i] == update_price) begin
                            bid_volume[i] <= update_volume;
                            found = 1'b1;
                        end
                    end

                    // Insert new level if not found and space available
                    if (!found && bid_count < DEPTH) begin
                        // Find insertion point (keep descending order)
                        ins_pos = bid_count[$clog2(DEPTH)-1:0];
                        for (i = 0; i < DEPTH-1; i = i + 1) begin
                            if (bid_valid[i] && bid_price[i] < update_price && i < ins_pos)
                                ins_pos = i[$clog2(DEPTH)-1:0];
                        end
                        // Shift levels down to make room
                        for (j = DEPTH-1; j > 0; j = j - 1) begin
                            if (j > ins_pos) begin
                                bid_price[j]  <= bid_price[j-1];
                                bid_volume[j] <= bid_volume[j-1];
                                bid_valid[j]  <= bid_valid[j-1];
                            end
                        end
                        bid_price[ins_pos]  <= update_price;
                        bid_volume[ins_pos] <= update_volume;
                        bid_valid[ins_pos]  <= 1'b1;
                        bid_count           <= bid_count + 1'b1;
                    end
                end
            end

        end else begin
            // ---- ASK side ----
            if (update_action == ACT_DELETE) begin
                for (i = 0; i < DEPTH; i = i + 1) begin
                    if (ask_valid[i] && ask_price[i] == update_price) begin
                        ask_valid[i] <= 1'b0;
                        if (ask_count > 0) ask_count <= ask_count - 1'b1;
                    end
                end
            end else begin
                begin : ask_upsert
                    reg found;
                    reg [$clog2(DEPTH)-1:0] ins_pos;
                    found   = 1'b0;
                    ins_pos = {$clog2(DEPTH){1'b0}};

                    for (i = 0; i < DEPTH; i = i + 1) begin
                        if (ask_valid[i] && ask_price[i] == update_price) begin
                            ask_volume[i] <= update_volume;
                            found = 1'b1;
                        end
                    end

                    if (!found && ask_count < DEPTH) begin
                        ins_pos = ask_count[$clog2(DEPTH)-1:0];
                        for (i = 0; i < DEPTH-1; i = i + 1) begin
                            if (ask_valid[i] && ask_price[i] > update_price && i < ins_pos)
                                ins_pos = i[$clog2(DEPTH)-1:0];
                        end
                        for (j = DEPTH-1; j > 0; j = j - 1) begin
                            if (j > ins_pos) begin
                                ask_price[j]  <= ask_price[j-1];
                                ask_volume[j] <= ask_volume[j-1];
                                ask_valid[j]  <= ask_valid[j-1];
                            end
                        end
                        ask_price[ins_pos]  <= update_price;
                        ask_volume[ins_pos] <= update_volume;
                        ask_valid[ins_pos]  <= 1'b1;
                        ask_count           <= ask_count + 1'b1;
                    end
                end
            end
        end

        // ---- Refresh best bid/ask ----
        // Use blocking assignments to compute next-state valid flags so the
        // scan sees deletions applied in this same cycle.
        begin : refresh_best
            reg [PRICE_WIDTH-1:0]  bb, ba;
            reg [VOLUME_WIDTH-1:0] bbq, baq;
            reg                    bv, av;
            reg                    nxt_bid_valid [0:DEPTH-1];
            reg                    nxt_ask_valid [0:DEPTH-1];
            integer k;

            // Compute next valid state (mirrors the delete logic above)
            for (k = 0; k < DEPTH; k = k + 1) begin
                nxt_bid_valid[k] = bid_valid[k];
                nxt_ask_valid[k] = ask_valid[k];
            end
            if (update_action == ACT_DELETE || update_action == ACT_CLEAR) begin
                if (update_action == ACT_CLEAR) begin
                    for (k = 0; k < DEPTH; k = k + 1) begin
                        if (!update_side) nxt_bid_valid[k] = 1'b0;
                        else              nxt_ask_valid[k] = 1'b0;
                    end
                end else begin
                    for (k = 0; k < DEPTH; k = k + 1) begin
                        if (!update_side && bid_price[k] == update_price)
                            nxt_bid_valid[k] = 1'b0;
                        if ( update_side && ask_price[k] == update_price)
                            nxt_ask_valid[k] = 1'b0;
                    end
                end
            end

            bb  = {PRICE_WIDTH{1'b0}};
            ba  = {PRICE_WIDTH{1'b1}};
            bbq = {VOLUME_WIDTH{1'b0}};
            baq = {VOLUME_WIDTH{1'b0}};
            bv  = 1'b0;
            av  = 1'b0;

            for (k = 0; k < DEPTH; k = k + 1) begin
                if (nxt_bid_valid[k] && bid_price[k] > bb) begin
                    bb  = bid_price[k];
                    bbq = bid_volume[k];
                    bv  = 1'b1;
                end
                if (nxt_ask_valid[k] && ask_price[k] < ba) begin
                    ba  = ask_price[k];
                    baq = ask_volume[k];
                    av  = 1'b1;
                end
            end

            // Also account for a fresh upsert in this cycle
            if (update_action == ACT_UPSERT) begin
                if (!update_side && update_price > bb) begin
                    bb  = update_price;
                    bbq = update_volume;
                    bv  = 1'b1;
                end
                if ( update_side && update_price < ba) begin
                    ba  = update_price;
                    baq = update_volume;
                    av  = 1'b1;
                end
            end

            best_bid     <= bb;
            best_bid_qty <= bbq;
            best_ask     <= ba;
            best_ask_qty <= baq;
            book_valid   <= bv && av;
        end
    end
end

endmodule
