`timescale 1ns/1ps

// Toxicity tracker approximating Eq. (6)-(7) in hjb_mm.pdf.
module hjb_toxicity_tracker #(
    parameter signed [31:0] ONE_Q16 = 32'sd65536,
    parameter signed [31:0] BETA_DECAY_Q16 = 32'sd58982, // exp(-0.1)
    parameter signed [31:0] ALPHA0_Q16 = 32'sd65536
) (
    input wire clk,
    input wire rst_n,
    input wire update_en,
    input wire signed [31:0] trade_dir_q16,      // +1/-1 in Q16
    input wire signed [31:0] spread_ticks,
    output reg signed [31:0] toxicity_q16,
    output reg signed [31:0] alpha_t_q16
);
    reg signed [63:0] weighted_dir_sum_q16;
    reg signed [63:0] weighted_spread_sum_q16;

    reg signed [63:0] next_dir_sum;
    reg signed [63:0] next_spread_sum;
    reg signed [63:0] toxicity_num;
    reg signed [31:0] toxicity_raw_q16;
    reg signed [31:0] spread_q16;
    reg signed [31:0] abs_tau_q16;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weighted_dir_sum_q16 <= 64'sd0;
            weighted_spread_sum_q16 <= 64'sd65536;
            toxicity_q16 <= 32'sd0;
            alpha_t_q16 <= ALPHA0_Q16;
        end else if (update_en) begin
            spread_q16 = (spread_ticks <= 0) ? ONE_Q16 : (spread_ticks <<< 16);

            next_dir_sum = ((weighted_dir_sum_q16 * BETA_DECAY_Q16) >>> 16) + trade_dir_q16;
            next_spread_sum = ((weighted_spread_sum_q16 * BETA_DECAY_Q16) >>> 16) + spread_q16;

            weighted_dir_sum_q16 <= next_dir_sum;
            weighted_spread_sum_q16 <= next_spread_sum;

            if (next_spread_sum <= 0)
                toxicity_raw_q16 = 32'sd0;
            else begin
                toxicity_num = next_dir_sum <<< 16;
                toxicity_raw_q16 = toxicity_num / next_spread_sum;
            end

            if (toxicity_raw_q16 > ONE_Q16)
                toxicity_q16 <= ONE_Q16;
            else if (toxicity_raw_q16 < -ONE_Q16)
                toxicity_q16 <= -ONE_Q16;
            else
                toxicity_q16 <= toxicity_raw_q16;

            abs_tau_q16 = (toxicity_raw_q16 < 0) ? -toxicity_raw_q16 : toxicity_raw_q16;
            if (abs_tau_q16 > ONE_Q16)
                abs_tau_q16 = ONE_Q16;

            // alpha_t = alpha_0 * (1 + 2 * |tau_t|)
            alpha_t_q16 <= (ALPHA0_Q16 * (ONE_Q16 + (abs_tau_q16 <<< 1))) >>> 16;
        end
    end
endmodule
