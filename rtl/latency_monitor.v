`timescale 1ns/1ps

/*
 * Latency Monitor
 * Centralized pipeline timing and SLA enforcement.
 *
 * Measures tick-to-order latency by timestamping incoming ticks and
 * correlating them with outgoing order events.  Maintains rolling
 * min/max/mean and a configurable SLA breach counter.
 *
 * Three independent measurement channels:
 *   CH0 - market_data_processor output → trading_strategy input
 *   CH1 - trading_strategy output      → order_manager input
 *   CH2 - order_manager input          → exec output  (full tick-to-trade)
 */

module latency_monitor #(
    parameter TIMESTAMP_WIDTH = 32,   // cycle counter width
    parameter LATENCY_WIDTH   = 32,
    parameter CHANNELS        = 3,
    parameter HISTORY_DEPTH   = 256   // rolling window for statistics
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // ---- channel start pulses ----
    input  wire [CHANNELS-1:0]          start_pulse,   // assert 1 cycle when event begins

    // ---- channel end pulses ----
    input  wire [CHANNELS-1:0]          end_pulse,     // assert 1 cycle when event ends

    // ---- SLA thresholds (cycles) ----
    input  wire [LATENCY_WIDTH-1:0]     sla_threshold_ch0,
    input  wire [LATENCY_WIDTH-1:0]     sla_threshold_ch1,
    input  wire [LATENCY_WIDTH-1:0]     sla_threshold_ch2,

    // ---- per-channel outputs ----
    output reg  [LATENCY_WIDTH-1:0]     last_latency   [0:CHANNELS-1],
    output reg  [LATENCY_WIDTH-1:0]     min_latency    [0:CHANNELS-1],
    output reg  [LATENCY_WIDTH-1:0]     max_latency    [0:CHANNELS-1],
    output reg  [LATENCY_WIDTH-1:0]     mean_latency   [0:CHANNELS-1],  // EMA, Q8 fixed point
    output reg  [31:0]                  sla_breach_cnt [0:CHANNELS-1],
    output reg  [31:0]                  sample_cnt     [0:CHANNELS-1],

    // ---- aggregate SLA breach flag ----
    output wire                         any_sla_breach,

    // ---- free-running cycle counter (exported for external timestamping) ----
    output wire [TIMESTAMP_WIDTH-1:0]   cycle_count
);

// ---------------------------------------------------------------------------
// Free-running counter
// ---------------------------------------------------------------------------
reg [TIMESTAMP_WIDTH-1:0] cycle_reg;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cycle_reg <= {TIMESTAMP_WIDTH{1'b0}};
    else        cycle_reg <= cycle_reg + 1'b1;
end
assign cycle_count = cycle_reg;

// ---------------------------------------------------------------------------
// Per-channel measurement
// ---------------------------------------------------------------------------
reg [TIMESTAMP_WIDTH-1:0] start_ts [0:CHANNELS-1];
reg                        pending  [0:CHANNELS-1];  // waiting for end pulse

// SLA threshold mux
reg [LATENCY_WIDTH-1:0] sla_thresh [0:CHANNELS-1];
always @(*) begin
    sla_thresh[0] = sla_threshold_ch0;
    sla_thresh[1] = sla_threshold_ch1;
    sla_thresh[2] = sla_threshold_ch2;
end

// EMA decay: mean = mean - (mean >> 3) + (sample >> 3)  (alpha ≈ 0.125)
localparam EMA_SHIFT = 3;

genvar ch;
generate
    for (ch = 0; ch < CHANNELS; ch = ch + 1) begin : channel_logic
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                start_ts[ch]      <= {TIMESTAMP_WIDTH{1'b0}};
                pending[ch]       <= 1'b0;
                last_latency[ch]  <= {LATENCY_WIDTH{1'b0}};
                min_latency[ch]   <= {LATENCY_WIDTH{1'b1}};  // max sentinel
                max_latency[ch]   <= {LATENCY_WIDTH{1'b0}};
                mean_latency[ch]  <= {LATENCY_WIDTH{1'b0}};
                sla_breach_cnt[ch]<= 32'b0;
                sample_cnt[ch]    <= 32'b0;
            end else begin
                // Capture start timestamp
                if (start_pulse[ch]) begin
                    start_ts[ch] <= cycle_reg;
                    pending[ch]  <= 1'b1;
                end

                // Measure on end pulse
                if (end_pulse[ch] && pending[ch]) begin
                    begin : measure
                        reg [LATENCY_WIDTH-1:0] lat;
                        lat = cycle_reg - start_ts[ch];

                        last_latency[ch] <= lat;
                        sample_cnt[ch]   <= sample_cnt[ch] + 1'b1;
                        pending[ch]      <= 1'b0;

                        // Min / max
                        if (lat < min_latency[ch]) min_latency[ch] <= lat;
                        if (lat > max_latency[ch]) max_latency[ch] <= lat;

                        // EMA mean
                        mean_latency[ch] <= mean_latency[ch]
                            - (mean_latency[ch] >> EMA_SHIFT)
                            + (lat             >> EMA_SHIFT);

                        // SLA breach
                        if (lat > sla_thresh[ch])
                            sla_breach_cnt[ch] <= sla_breach_cnt[ch] + 1'b1;
                    end
                end
            end
        end
    end
endgenerate

// ---------------------------------------------------------------------------
// Aggregate breach flag
// ---------------------------------------------------------------------------
assign any_sla_breach = (sla_breach_cnt[0] != 32'b0) ||
                        (sla_breach_cnt[1] != 32'b0) ||
                        (sla_breach_cnt[2] != 32'b0);

endmodule
