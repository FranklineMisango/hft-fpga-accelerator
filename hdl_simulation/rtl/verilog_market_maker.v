`timescale 1ns/1ps

// Verilog market-maker quote core
module verilog_market_maker #(
    parameter integer BASE_SPREAD_TICKS = 4,
    parameter integer REPLACE_THRESHOLD_TICKS = 2
) (
    input wire clk,
    input wire rst_n,
    input wire calculate_en,
    input wire signed [31:0] mid_ticks,
    input wire signed [31:0] best_bid_ticks,
    input wire signed [31:0] best_ask_ticks,
    input wire signed [31:0] inventory_milli,
    input wire signed [31:0] volatility_bp,
    input wire signed [31:0] bid_qty_milli,
    input wire signed [31:0] ask_qty_milli,
    output reg signed [31:0] quote_bid_ticks,
    output reg signed [31:0] quote_ask_ticks,
    output reg replace_hint,
    output reg cancel_hint,
    output reg done,
    output reg [31:0] latency_cycles
);

    localparam [1:0] IDLE = 2'b00;
    localparam [1:0] COMPUTE = 2'b01;
    localparam [1:0] OUTPUT_STATE = 2'b10;

    localparam signed [31:0] ONE_Q16 = 32'sd65536;
    localparam signed [31:0] A_B_Q16 = 32'sd65536;
    localparam signed [31:0] A_A_Q16 = 32'sd65536;
    localparam signed [31:0] GAMMA_INV_Q20 = 32'sd32768; // 0.03125
    localparam signed [31:0] KAPPA_Q20 = 32'sd1049;      // ~0.001

    reg [1:0] state;
    reg signed [31:0] prev_bid;
    reg signed [31:0] prev_ask;

    reg signed [31:0] spread_ticks;
    reg signed [31:0] half_spread;
    reg signed [31:0] reservation;
    reg signed [31:0] target_bid;
    reg signed [31:0] target_ask;
    reg signed [31:0] bid_diff;
    reg signed [31:0] ask_diff;

    reg toxicity_update_en;
    reg signed [31:0] trade_dir_q16;
    reg signed [63:0] best_score;
    reg signed [63:0] score_candidate;
    reg signed [63:0] expected_pnl;
    reg signed [63:0] inventory_cost;
    reg signed [63:0] inv_sq;
    reg signed [63:0] vol_sq;
    reg signed [63:0] inv_skew;

    reg signed [31:0] cand_bid;
    reg signed [31:0] cand_ask;
    reg signed [31:0] best_bid_local;
    reg signed [31:0] best_ask_local;
    reg signed [31:0] lambda_b_q16;
    reg signed [31:0] lambda_a_q16;

    integer bi;
    integer ai;

    wire signed [31:0] toxicity_q16;
    wire signed [31:0] alpha_t_q16;
    wire signed [31:0] jump_drift_ticks;
    wire signed [31:0] jump_var_q20;

    hjb_toxicity_tracker toxicity_tracker (
        .clk(clk),
        .rst_n(rst_n),
        .update_en(toxicity_update_en),
        .trade_dir_q16(trade_dir_q16),
        .spread_ticks(spread_ticks),
        .toxicity_q16(toxicity_q16),
        .alpha_t_q16(alpha_t_q16)
    );

    hjb_jump_operator jump_operator (
        .mid_ticks(mid_ticks),
        .jump_drift_ticks(jump_drift_ticks),
        .jump_var_q20(jump_var_q20)
    );

    function signed [31:0] abs32;
        input signed [31:0] value;
        begin
            if (value < 0)
                abs32 = -value;
            else
                abs32 = value;
        end
    endfunction

    function signed [31:0] intensity_q16;
        input signed [31:0] quote_ticks;
        input signed [31:0] market_ticks;
        input signed [31:0] baseline_q16;
        input signed [31:0] alpha_q16;
        reg signed [63:0] ratio_minus_one_q16;
        reg signed [63:0] delta_over_alpha_q16;
        reg signed [63:0] shape_q16;
        begin
            if ((market_ticks <= 0) || (alpha_q16 <= 0)) begin
                intensity_q16 = 32'sd0;
            end else begin
                ratio_minus_one_q16 = ((quote_ticks - market_ticks) <<< 16) / market_ticks;
                delta_over_alpha_q16 = (ratio_minus_one_q16 <<< 16) / alpha_q16;
                shape_q16 = ONE_Q16 - delta_over_alpha_q16;
                if (shape_q16 <= 0)
                    intensity_q16 = 32'sd0;
                else
                    intensity_q16 = (baseline_q16 * shape_q16) >>> 16;
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            quote_bid_ticks <= 32'sd0;
            quote_ask_ticks <= 32'sd0;
            prev_bid <= 32'sd0;
            prev_ask <= 32'sd0;
            replace_hint <= 1'b0;
            cancel_hint <= 1'b0;
            done <= 1'b0;
            latency_cycles <= 32'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    replace_hint <= 1'b0;
                    cancel_hint <= 1'b0;
                    latency_cycles <= 32'd0;
                    toxicity_update_en <= 1'b0;
                    if (calculate_en) begin
                        state <= COMPUTE;
                        latency_cycles <= latency_cycles + 1;
                    end
                end

                COMPUTE: begin
                    spread_ticks = best_ask_ticks - best_bid_ticks;
                    if (spread_ticks < 32'sd1)
                        spread_ticks = 32'sd1;

                    trade_dir_q16 = (bid_qty_milli >= ask_qty_milli) ? ONE_Q16 : -ONE_Q16;
                    toxicity_update_en <= 1'b1;

                    vol_sq = volatility_bp * volatility_bp;
                    inv_skew = (inventory_milli * GAMMA_INV_Q20 * vol_sq) >>> 20;

                    reservation = mid_ticks + jump_drift_ticks - inv_skew[31:0];
                    half_spread = BASE_SPREAD_TICKS + (volatility_bp >>> 5) + (jump_var_q20 >>> 18);
                    half_spread = (half_spread * alpha_t_q16) >>> 16;
                    if (half_spread < 32'sd1)
                        half_spread = 32'sd1;

                    best_score = -64'sh3FFF_FFFF_FFFF_FFFF;
                    best_bid_local = best_bid_ticks;
                    best_ask_local = best_ask_ticks;

                    // 5x5 control-space search, matching the paper's finite control optimization.
                    for (bi = 0; bi < 5; bi = bi + 1) begin
                        cand_bid = reservation - half_spread + (bi - 2);
                        if (cand_bid > best_bid_ticks)
                            cand_bid = best_bid_ticks;
                        if (cand_bid < 32'sd1)
                            cand_bid = 32'sd1;

                        for (ai = 0; ai < 5; ai = ai + 1) begin
                            cand_ask = reservation + half_spread + (ai - 2);
                            if (cand_ask < best_ask_ticks)
                                cand_ask = best_ask_ticks;
                            if (cand_ask <= cand_bid)
                                cand_ask = cand_bid + 32'sd1;

                            lambda_b_q16 = intensity_q16(cand_bid, best_bid_ticks, A_B_Q16, alpha_t_q16);
                            lambda_a_q16 = intensity_q16(cand_ask, best_ask_ticks, A_A_Q16, alpha_t_q16);

                            expected_pnl = (cand_ask * lambda_a_q16) - (cand_bid * lambda_b_q16);

                            inv_sq = inventory_milli * inventory_milli;
                            inventory_cost = (KAPPA_Q20 * inv_sq) >>> 4;

                            score_candidate = expected_pnl - inventory_cost
                                - ((abs32(cand_bid - (reservation - half_spread))
                                + abs32(cand_ask - (reservation + half_spread))) <<< 12)
                                + (vol_sq <<< 8);

                            if (score_candidate > best_score) begin
                                best_score = score_candidate;
                                best_bid_local = cand_bid;
                                best_ask_local = cand_ask;
                            end
                        end
                    end

                    target_bid = best_bid_local;
                    target_ask = best_ask_local;

                    quote_bid_ticks <= target_bid;
                    quote_ask_ticks <= target_ask;
                    latency_cycles <= latency_cycles + 1;
                    state <= OUTPUT_STATE;
                end

                OUTPUT_STATE: begin
                    toxicity_update_en <= 1'b0;
                    bid_diff = abs32(quote_bid_ticks - prev_bid);
                    ask_diff = abs32(quote_ask_ticks - prev_ask);

                    if ((bid_diff >= REPLACE_THRESHOLD_TICKS) || (ask_diff >= REPLACE_THRESHOLD_TICKS))
                        replace_hint <= 1'b1;
                    else
                        replace_hint <= 1'b0;

                    if (quote_bid_ticks >= quote_ask_ticks)
                        cancel_hint <= 1'b1;
                    else
                        cancel_hint <= 1'b0;

                    prev_bid <= quote_bid_ticks;
                    prev_ask <= quote_ask_ticks;
                    done <= 1'b1;
                    latency_cycles <= latency_cycles + 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
