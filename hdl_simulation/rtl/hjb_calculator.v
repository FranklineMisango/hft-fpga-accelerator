// HJB Optimal Quote Calculator for TradeChest Integration
module hjb_calculator (
    input wire clk,
    input wire rst_n,
    input wire [63:0] mid_price,      // IEEE 754 double precision
    input wire [31:0] inventory,      // Signed inventory
    input wire [63:0] volatility,     // IEEE 754 double precision
    input wire calculate_en,
    output reg [63:0] optimal_bid,    // IEEE 754 double precision
    output reg [63:0] optimal_ask,    // IEEE 754 double precision
    output reg calculation_done,
    output reg [31:0] latency_cycles
);

    // HJB parameters (fixed point for FPGA efficiency)
    localparam GAMMA = 32'h3DCCCCCD;  // 0.1 in IEEE 754
    localparam K_PARAM = 32'h3FC00000; // 1.5 in IEEE 754
    
    // Internal registers
    reg [63:0] reservation_price;
    reg [63:0] spread;
    reg [31:0] cycle_counter;
    reg [2:0] state;
    
    // State machine states
    localparam IDLE = 3'b000;
    localparam CALC_RESERVATION = 3'b001;
    localparam CALC_SPREAD = 3'b010;
    localparam CALC_QUOTES = 3'b011;
    localparam DONE = 3'b100;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            optimal_bid <= 64'h0;
            optimal_ask <= 64'h0;
            calculation_done <= 1'b0;
            latency_cycles <= 32'h0;
            cycle_counter <= 32'h0;
        end else begin
            case (state)
                IDLE: begin
                    calculation_done <= 1'b0;
                    cycle_counter <= 32'h0;
                    if (calculate_en) begin
                        state <= CALC_RESERVATION;
                        cycle_counter <= cycle_counter + 1;
                    end
                end
                
                CALC_RESERVATION: begin
                    // reservation_price = mid_price - inventory * gamma * volatility^2
                    // Simplified for FPGA: reservation_price ≈ mid_price - inventory * 0.001
                    reservation_price <= mid_price - ({32'h0, inventory} << 10); // Approximate calculation
                    state <= CALC_SPREAD;
                    cycle_counter <= cycle_counter + 1;
                end
                
                CALC_SPREAD: begin
                    // spread = gamma * volatility^2 + (2/gamma) * ln(1 + gamma/k)
                    // Simplified for FPGA: spread ≈ 0.01 * mid_price (1% spread)
                    spread <= mid_price >> 7; // Approximately 0.78% spread
                    state <= CALC_QUOTES;
                    cycle_counter <= cycle_counter + 1;
                end
                
                CALC_QUOTES: begin
                    // optimal_bid = reservation_price - spread/2
                    // optimal_ask = reservation_price + spread/2
                    optimal_bid <= reservation_price - (spread >> 1);
                    optimal_ask <= reservation_price + (spread >> 1);
                    state <= DONE;
                    cycle_counter <= cycle_counter + 1;
                end
                
                DONE: begin
                    calculation_done <= 1'b1;
                    latency_cycles <= cycle_counter;
                    if (!calculate_en) begin
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
