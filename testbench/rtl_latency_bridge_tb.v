`timescale 1ns/1ps
// Bridge harness for latency_monitor.
// Reads one line from rtl_latency_input.txt:
//   <channel>,<latency_cycles>,<sla_threshold>
// Writes one line to rtl_latency_output.txt:
//   <last_latency_ch0>,<min_latency_ch0>,<max_latency_ch0>,<mean_latency_ch0>,
//   <sla_breach_cnt_ch0>,<sample_cnt_ch0>,<any_sla_breach>

module rtl_latency_bridge_tb;

    reg clk, rst_n;
    reg [2:0] start_pulse, end_pulse;
    reg [31:0] sla_ch0, sla_ch1, sla_ch2;

    wire [31:0] last_latency[0:2];
    wire [31:0] min_latency[0:2];
    wire [31:0] max_latency[0:2];
    wire [31:0] mean_latency[0:2];
    wire [31:0] sla_breach_cnt[0:2];
    wire [31:0] sample_cnt[0:2];
    wire        any_sla_breach;
    wire [31:0] cycle_count;

    latency_monitor dut (
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

    integer fv, fo, k;
    integer r_channel, r_latency, r_sla;

    initial begin
        clk = 0; rst_n = 0;
        start_pulse = 3'b0; end_pulse = 3'b0;
        sla_ch0 = 32'd100; sla_ch1 = 32'd100; sla_ch2 = 32'd100;
        @(posedge clk); #1; rst_n = 1;
        @(posedge clk); #1;

        fv = $fopen("rtl_latency_input.txt", "r");
        if (fv == 0) begin $display("ERROR: cannot open rtl_latency_input.txt"); $finish; end
        r_channel = 0; r_latency = 0; r_sla = 100;
        $fscanf(fv, "%d,%d,%d\n", r_channel, r_latency, r_sla);
        $fclose(fv);

        sla_ch0 = r_sla; sla_ch1 = r_sla; sla_ch2 = r_sla;

        // Assert start on requested channel
        start_pulse = (3'b001 << r_channel[1:0]);
        @(posedge clk); #1;
        start_pulse = 3'b0;

        // Wait r_latency cycles
        for (k = 0; k < r_latency - 1; k = k + 1)
            @(posedge clk);
        #1;

        // Assert end
        end_pulse = (3'b001 << r_channel[1:0]);
        @(posedge clk); #1;
        end_pulse = 3'b0;
        repeat(3) @(posedge clk); #1;

        fo = $fopen("rtl_latency_output.txt", "w");
        $fdisplay(fo, "%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            last_latency[0], min_latency[0], max_latency[0], mean_latency[0],
            sla_breach_cnt[0], sample_cnt[0], any_sla_breach);
        $fclose(fo);
        $finish;
    end
endmodule
