`timescale 1ns/1ps

module latency_monitor_tb;

    localparam TIMESTAMP_WIDTH = 32;
    localparam LATENCY_WIDTH   = 32;
    localparam CHANNELS        = 3;

    reg clk, rst_n;
    reg [CHANNELS-1:0] start_pulse, end_pulse;
    reg [LATENCY_WIDTH-1:0] sla_ch0, sla_ch1, sla_ch2;

    wire [LATENCY_WIDTH-1:0] last_latency   [0:CHANNELS-1];
    wire [LATENCY_WIDTH-1:0] min_latency    [0:CHANNELS-1];
    wire [LATENCY_WIDTH-1:0] max_latency    [0:CHANNELS-1];
    wire [LATENCY_WIDTH-1:0] mean_latency   [0:CHANNELS-1];
    wire [31:0]              sla_breach_cnt [0:CHANNELS-1];
    wire [31:0]              sample_cnt     [0:CHANNELS-1];
    wire                     any_sla_breach;
    wire [TIMESTAMP_WIDTH-1:0] cycle_count;

    latency_monitor #(
        .TIMESTAMP_WIDTH(TIMESTAMP_WIDTH),
        .LATENCY_WIDTH(LATENCY_WIDTH),
        .CHANNELS(CHANNELS)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start_pulse(start_pulse), .end_pulse(end_pulse),
        .sla_threshold_ch0(sla_ch0),
        .sla_threshold_ch1(sla_ch1),
        .sla_threshold_ch2(sla_ch2),
        .last_latency(last_latency),
        .min_latency(min_latency),
        .max_latency(max_latency),
        .mean_latency(mean_latency),
        .sla_breach_cnt(sla_breach_cnt),
        .sample_cnt(sample_cnt),
        .any_sla_breach(any_sla_breach),
        .cycle_count(cycle_count)
    );

    always #5 clk = ~clk;

    task measure_latency;
        input integer ch;
        input integer delay_cycles;
        integer k;
        begin
            start_pulse = (1 << ch);
            @(posedge clk); #1;
            start_pulse = 3'b0;
            for (k = 0; k < delay_cycles - 1; k = k + 1)
                @(posedge clk);
            #1;
            end_pulse = (1 << ch);
            @(posedge clk); #1;
            end_pulse = 3'b0;
            @(posedge clk); #1;
        end
    endtask

    integer pass_count, fail_count;

    initial begin
        clk = 0; rst_n = 0; pass_count = 0; fail_count = 0;
        start_pulse = 3'b0; end_pulse = 3'b0;
        sla_ch0 = 32'd20; sla_ch1 = 32'd20; sla_ch2 = 32'd20;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        // Test 1: Measure 10-cycle latency on CH0
        measure_latency(0, 10);
        if (last_latency[0] >= 32'd9 && last_latency[0] <= 32'd11) begin
            $display("PASS: CH0 latency ~10 cycles = %0d", last_latency[0]); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: CH0 latency unexpected = %0d", last_latency[0]); fail_count = fail_count + 1;
        end

        // Test 2: No SLA breach for 10-cycle (threshold=20)
        if (sla_breach_cnt[0] == 32'd0) begin
            $display("PASS: No SLA breach for 10-cycle latency"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Unexpected SLA breach"); fail_count = fail_count + 1;
        end

        // Test 3: Trigger SLA breach on CH1 (30 cycles > threshold 20)
        measure_latency(1, 30);
        if (sla_breach_cnt[1] == 32'd1) begin
            $display("PASS: SLA breach detected on CH1"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: SLA breach not detected (cnt=%0d)", sla_breach_cnt[1]); fail_count = fail_count + 1;
        end

        // Test 4: any_sla_breach should be high
        if (any_sla_breach) begin
            $display("PASS: any_sla_breach asserted"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: any_sla_breach not asserted"); fail_count = fail_count + 1;
        end

        // Test 5: Sample count increments
        measure_latency(0, 5);
        if (sample_cnt[0] == 32'd2) begin
            $display("PASS: Sample count = %0d", sample_cnt[0]); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Expected sample_cnt 2, got %0d", sample_cnt[0]); fail_count = fail_count + 1;
        end

        // Test 6: Min latency tracks correctly
        if (min_latency[0] <= last_latency[0]) begin
            $display("PASS: Min latency tracking correct"); pass_count = pass_count + 1;
        end else begin
            $display("FAIL: Min latency incorrect"); fail_count = fail_count + 1;
        end

        $display("---");
        $display("latency_monitor_tb: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0) $display("PASS: All latency_monitor tests passed");
        else                 $display("FAIL: Some latency_monitor tests failed");
        $finish;
    end

endmodule
