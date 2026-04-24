`timescale 1ns/1ps

// 5-point Gauss-Hermite jump approximation (Algorithm 3 in hjb_mm.pdf).
module hjb_jump_operator #(
    parameter signed [31:0] ONE_Q20 = 32'sd1048576,
    parameter signed [31:0] LAMBDA_J_Q20 = 32'sd209715, // 0.20
    parameter signed [31:0] MU_J_Q20 = 32'sd0,
    parameter signed [31:0] SIGMA_J_Q20 = 32'sd41943 // 0.04
) (
    input wire signed [31:0] mid_ticks,
    output reg signed [31:0] jump_drift_ticks,
    output reg signed [31:0] jump_var_q20
);
    integer k;
    reg signed [31:0] z_q16 [0:4];
    reg signed [31:0] w_q16 [0:4];
    reg signed [63:0] sum_weighted_y_q20;
    reg signed [63:0] yk_q20;
    reg signed [63:0] lambda_times_sum_q20;
    reg signed [63:0] drift_q20;

    initial begin
        z_q16[0] = -32'sd132395; // -2.02018
        z_q16[1] = -32'sd62821;  // -0.95857
        z_q16[2] = 32'sd0;
        z_q16[3] = 32'sd62821;   // 0.95857
        z_q16[4] = 32'sd132395;  // 2.02018

        w_q16[0] = 32'sd5783;    // 0.08824
        w_q16[1] = 32'sd25796;   // 0.39362
        w_q16[2] = 32'sd61952;   // 0.94531
        w_q16[3] = 32'sd25796;   // 0.39362
        w_q16[4] = 32'sd5783;    // 0.08824
    end

    always @(*) begin
        sum_weighted_y_q20 = 64'sd0;

        for (k = 0; k < 5; k = k + 1) begin
            yk_q20 = MU_J_Q20 + ((SIGMA_J_Q20 * z_q16[k]) >>> 16);
            sum_weighted_y_q20 = sum_weighted_y_q20 + ((w_q16[k] * yk_q20) >>> 16);
        end

        lambda_times_sum_q20 = (LAMBDA_J_Q20 * sum_weighted_y_q20) >>> 20;
        drift_q20 = mid_ticks * lambda_times_sum_q20;
        jump_drift_ticks = drift_q20 >>> 20;

        // Variance uplift proxy used by spread controller.
        jump_var_q20 = (LAMBDA_J_Q20 * (MU_J_Q20 * MU_J_Q20 + SIGMA_J_Q20 * SIGMA_J_Q20)) >>> 20;
    end
endmodule
