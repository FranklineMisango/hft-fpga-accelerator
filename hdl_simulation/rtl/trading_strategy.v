/*
 * Trading Strategy Engine
 * Hardware-accelerated trading strategy execution
 * 
 * Features:
 * - Arbitrage detection
 * - Market making
 * - TWAP execution
 * - Sub-microsecond decision making
 */

module trading_strategy #(
    parameter SYMBOL_WIDTH = 32,
    parameter PRICE_WIDTH = 32,
    parameter VOLUME_WIDTH = 32,
    parameter MAX_SYMBOLS = 256,
    parameter STRATEGY_COUNT = 4
) (
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Market data input
    input  wire                     tick_valid,
    input  wire [SYMBOL_WIDTH-1:0]  tick_symbol,
    input  wire [PRICE_WIDTH-1:0]   tick_price,
    input  wire [PRICE_WIDTH-1:0]   tick_bid,
    input  wire [PRICE_WIDTH-1:0]   tick_ask,
    input  wire [VOLUME_WIDTH-1:0]  tick_volume,
    
    // Strategy configuration
    input  wire [3:0]               strategy_enable,    // Enable bits for each strategy
    input  wire [PRICE_WIDTH-1:0]   arb_min_profit,    // Minimum profit for arbitrage
    input  wire [PRICE_WIDTH-1:0]   mm_spread,         // Market making spread
    input  wire [VOLUME_WIDTH-1:0]  twap_target_vol,   // TWAP target volume
    input  wire [31:0]              twap_duration,     // TWAP duration in cycles
    
    // Order generation output
    output reg                      order_valid,
    output reg  [SYMBOL_WIDTH-1:0]  order_symbol,
    output reg  [PRICE_WIDTH-1:0]   order_price,
    output reg  [VOLUME_WIDTH-1:0]  order_volume,
    output reg                      order_side,        // 0=buy, 1=sell
    output reg  [2:0]               order_type,        // 0=market, 1=limit
    
    // Position interface
    input  wire [VOLUME_WIDTH-1:0]  current_position,
    input  wire [PRICE_WIDTH-1:0]   position_limit,
    
    // Performance metrics
    output wire [31:0]              decisions_made,
    output wire [31:0]              orders_generated,
    output wire [15:0]              active_strategies
);

// Strategy types
localparam STRATEGY_ARBITRAGE = 2'b00;
localparam STRATEGY_MARKET_MAKING = 2'b01;
localparam STRATEGY_TWAP = 2'b10;
localparam STRATEGY_MOMENTUM = 2'b11;

// Internal registers
reg [31:0] decision_counter;
reg [31:0] order_gen_counter;
reg [15:0] active_strategy_mask;

// Price history for strategies
reg [PRICE_WIDTH-1:0] price_history [0:15];  // 16-element circular buffer
reg [3:0] price_history_ptr;
reg [PRICE_WIDTH-1:0] last_price;

// Arbitrage detection
reg [PRICE_WIDTH-1:0] arb_price_1, arb_price_2;
reg [SYMBOL_WIDTH-1:0] arb_symbol_1, arb_symbol_2;
wire arb_opportunity;
wire [PRICE_WIDTH-1:0] arb_profit;

// Market making state
reg [PRICE_WIDTH-1:0] mm_bid_price, mm_ask_price;
reg [VOLUME_WIDTH-1:0] mm_bid_volume, mm_ask_volume;
reg mm_quote_valid;
reg [1:0] mm_state;

// TWAP execution
reg [31:0] twap_timer;
reg [VOLUME_WIDTH-1:0] twap_executed;
reg [VOLUME_WIDTH-1:0] twap_slice_size;
reg twap_active;
wire twap_time_slice;

// Strategy decision logic
reg [1:0] selected_strategy;
reg strategy_decision_valid;

// Pipeline registers
reg tick_valid_d1, tick_valid_d2;
reg [SYMBOL_WIDTH-1:0] tick_symbol_d1, tick_symbol_d2;
reg [PRICE_WIDTH-1:0] tick_price_d1, tick_price_d2;
reg [PRICE_WIDTH-1:0] tick_bid_d1, tick_bid_d2;
reg [PRICE_WIDTH-1:0] tick_ask_d1, tick_ask_d2;

// Statistics
assign decisions_made = decision_counter;
assign orders_generated = order_gen_counter;
assign active_strategies = active_strategy_mask;

// Main strategy execution
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset all registers
        decision_counter <= 32'b0;
        order_gen_counter <= 32'b0;
        active_strategy_mask <= 16'b0;
        price_history_ptr <= 4'b0;
        last_price <= {PRICE_WIDTH{1'b0}};
        
        // Reset strategy states
        mm_state <= 2'b00;
        mm_quote_valid <= 1'b0;
        twap_timer <= 32'b0;
        twap_executed <= {VOLUME_WIDTH{1'b0}};
        twap_active <= 1'b0;
        
        // Reset pipeline
        tick_valid_d1 <= 1'b0;
        tick_valid_d2 <= 1'b0;
        
        order_valid <= 1'b0;
        strategy_decision_valid <= 1'b0;
        
        // Initialize price history
        integer i;
        for (i = 0; i < 16; i = i + 1) begin
            price_history[i] <= {PRICE_WIDTH{1'b0}};
        end
        
    end else begin
        // Pipeline advancement
        tick_valid_d1 <= tick_valid;
        tick_valid_d2 <= tick_valid_d1;
        tick_symbol_d1 <= tick_symbol;
        tick_symbol_d2 <= tick_symbol_d1;
        tick_price_d1 <= tick_price;
        tick_price_d2 <= tick_price_d1;
        tick_bid_d1 <= tick_bid;
        tick_bid_d2 <= tick_bid_d1;
        tick_ask_d1 <= tick_ask;
        tick_ask_d2 <= tick_ask_d1;
        
        // Update price history
        if (tick_valid) begin
            price_history[price_history_ptr] <= tick_price;
            price_history_ptr <= price_history_ptr + 1;
            last_price <= tick_price;
        end
        
        // Strategy execution pipeline
        order_valid <= 1'b0;
        strategy_decision_valid <= 1'b0;
        
        if (tick_valid_d2) begin
            decision_counter <= decision_counter + 1;
            
            // Strategy selection based on market conditions
            if (strategy_enable[0] && arb_opportunity) begin
                selected_strategy <= STRATEGY_ARBITRAGE;
                strategy_decision_valid <= 1'b1;
            end else if (strategy_enable[1] && mm_quote_valid) begin
                selected_strategy <= STRATEGY_MARKET_MAKING;
                strategy_decision_valid <= 1'b1;
            end else if (strategy_enable[2] && twap_active && twap_time_slice) begin
                selected_strategy <= STRATEGY_TWAP;
                strategy_decision_valid <= 1'b1;
            end else if (strategy_enable[3] && momentum_signal) begin
                selected_strategy <= STRATEGY_MOMENTUM;
                strategy_decision_valid <= 1'b1;
            end
        end
        
        // Execute selected strategy
        if (strategy_decision_valid) begin
            case (selected_strategy)
                STRATEGY_ARBITRAGE: begin
                    execute_arbitrage();
                end
                
                STRATEGY_MARKET_MAKING: begin
                    execute_market_making();
                end
                
                STRATEGY_TWAP: begin
                    execute_twap();
                end
                
                STRATEGY_MOMENTUM: begin
                    execute_momentum();
                end
            endcase
        end
        
        // Update strategy states
        update_arbitrage_state();
        update_market_making_state();
        update_twap_state();
        
        // Update active strategies mask
        active_strategy_mask <= {12'b0, strategy_enable};
    end
end

// Arbitrage detection logic
assign arb_profit = (tick_ask_d2 > tick_bid_d2) ? (tick_ask_d2 - tick_bid_d2) : 32'b0;
assign arb_opportunity = (arb_profit > arb_min_profit) && (current_position < position_limit);

// Market making time slice
assign twap_time_slice = (twap_timer > 0) && ((twap_timer % (twap_duration / 100)) == 0);

// Momentum signal (simplified)
wire [PRICE_WIDTH-1:0] price_change;
wire momentum_signal;
assign price_change = (tick_price_d2 > last_price) ? (tick_price_d2 - last_price) : 
                     (last_price - tick_price_d2);
assign momentum_signal = (price_change > (tick_price_d2 >> 7));  // 0.78% threshold

// Strategy execution tasks
task execute_arbitrage;
    begin
        if (arb_profit > arb_min_profit) begin
            order_valid <= 1'b1;
            order_symbol <= tick_symbol_d2;
            order_price <= tick_bid_d2;
            order_volume <= 32'd1000;  // Fixed size for simplicity
            order_side <= 1'b0;  // Buy
            order_type <= 3'b001;  // Limit order
            
            order_gen_counter <= order_gen_counter + 1;
        end
    end
endtask

task execute_market_making;
    begin
        if (mm_quote_valid) begin
            // Generate bid order
            order_valid <= 1'b1;
            order_symbol <= tick_symbol_d2;
            order_price <= mm_bid_price;
            order_volume <= mm_bid_volume;
            order_side <= 1'b0;  // Buy
            order_type <= 3'b001;  // Limit order
            
            order_gen_counter <= order_gen_counter + 1;
        end
    end
endtask

task execute_twap;
    begin
        if (twap_active && (twap_executed < twap_target_vol)) begin
            order_valid <= 1'b1;
            order_symbol <= tick_symbol_d2;
            order_price <= tick_price_d2;
            order_volume <= twap_slice_size;
            order_side <= 1'b0;  // Buy (could be parameterized)
            order_type <= 3'b000;  // Market order
            
            twap_executed <= twap_executed + twap_slice_size;
            order_gen_counter <= order_gen_counter + 1;
        end
    end
endtask

task execute_momentum;
    begin
        if (momentum_signal) begin
            order_valid <= 1'b1;
            order_symbol <= tick_symbol_d2;
            order_price <= tick_price_d2;
            order_volume <= 32'd500;  // Fixed size
            order_side <= (price_change > 0) ? 1'b0 : 1'b1;  // Buy on up move, sell on down
            order_type <= 3'b000;  // Market order
            
            order_gen_counter <= order_gen_counter + 1;
        end
    end
endtask

// Strategy state update tasks
task update_arbitrage_state;
    begin
        // Store prices for cross-symbol arbitrage
        arb_price_1 <= tick_price_d1;
        arb_symbol_1 <= tick_symbol_d1;
        arb_price_2 <= tick_price_d2;
        arb_symbol_2 <= tick_symbol_d2;
    end
endtask

task update_market_making_state;
    begin
        if (tick_valid_d1) begin
            // Calculate market making quotes
            mm_bid_price <= tick_bid_d1 + (mm_spread >> 1);
            mm_ask_price <= tick_ask_d1 - (mm_spread >> 1);
            mm_bid_volume <= 32'd1000;
            mm_ask_volume <= 32'd1000;
            mm_quote_valid <= 1'b1;
        end
    end
endtask

task update_twap_state;
    begin
        if (twap_active) begin
            twap_timer <= twap_timer + 1;
            
            // Calculate slice size
            twap_slice_size <= (twap_target_vol - twap_executed) / 
                              ((twap_duration - twap_timer) / 100 + 1);
            
            // Check if TWAP is complete
            if (twap_executed >= twap_target_vol || twap_timer >= twap_duration) begin
                twap_active <= 1'b0;
                twap_timer <= 32'b0;
                twap_executed <= {VOLUME_WIDTH{1'b0}};
            end
        end else if (strategy_enable[2] && twap_target_vol > 0) begin
            // Start TWAP if enabled and target volume is set
            twap_active <= 1'b1;
            twap_timer <= 32'b0;
            twap_executed <= {VOLUME_WIDTH{1'b0}};
        end
    end
endtask

endmodule
