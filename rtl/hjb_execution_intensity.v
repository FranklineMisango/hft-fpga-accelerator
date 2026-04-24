`timescale 1ns/1ps

// Piecewise linear execution intensity (Eq. 4-5 in hjb_mm.pdf).
module hjb_execution_intensity #(
    parameter signed [31:0] ONE_Q16 = 32'sd65536
) (
    input wire signed [31:0] quote_ticks,
    input wire signed [31:0] market_ticks,
    input wire signed [31:0] baseline_intensity_q16,
    input wire signed [31:0] alpha_t_q16,
    output reg signed [31:0] intensity_q16
);
    reg signed [63:0] ratio_minus_one_q16;
    reg signed [63:0] delta_over_alpha_q16;
    reg signed [63:0] shape_q16;

    always @(*) begin
        if (market_ticks <= 0 || alpha_t_q16 <= 0) begin
            intensity_q16 = 32'sd0;
        end else begin
            ratio_minus_one_q16 = ((quote_ticks - market_ticks) <<< 16) / market_ticks;
            delta_over_alpha_q16 = (ratio_minus_one_q16 <<< 16) / alpha_t_q16;
            shape_q16 = ONE_Q16 - delta_over_alpha_q16;
            if (shape_q16 <= 0)
                intensity_q16 = 32'sd0;
            else
                intensity_q16 = (baseline_intensity_q16 * shape_q16) >>> 16;
        end
    end
endmodule
