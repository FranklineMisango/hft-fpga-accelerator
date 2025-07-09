/*
 * Market Data Processing Module
 * Hardware-accelerated market data parsing and order book management
 * 
 * Features:
 * - ITCH/FIX protocol parsing
 * - Real-time order book updates
 * - Sub-microsecond latency
 * - Pipelined architecture
 */

module market_data_processor #(
    parameter DATA_WIDTH = 64,
    parameter ADDR_WIDTH = 32,
    parameter SYMBOL_WIDTH = 32,
    parameter PRICE_WIDTH = 32,
    parameter VOLUME_WIDTH = 32,
    parameter MAX_ORDERS = 1024,
    parameter MAX_SYMBOLS = 256
) (
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Input market data stream
    input  wire                     data_valid,
    input  wire [DATA_WIDTH-1:0]    data_in,
    input  wire [7:0]               data_type,      // ITCH message type
    output wire                     data_ready,
    
    // Parsed market data output
    output reg                      tick_valid,
    output reg  [SYMBOL_WIDTH-1:0]  symbol,
    output reg  [PRICE_WIDTH-1:0]   price,
    output reg  [VOLUME_WIDTH-1:0]  volume,
    output reg  [PRICE_WIDTH-1:0]   bid,
    output reg  [PRICE_WIDTH-1:0]   ask,
    output reg  [63:0]              timestamp,
    
    // Order book interface
    output reg                      book_update_valid,
    output reg  [SYMBOL_WIDTH-1:0]  book_symbol,
    output reg  [PRICE_WIDTH-1:0]   book_price,
    output reg  [VOLUME_WIDTH-1:0]  book_volume,
    output reg                      book_side,      // 0=buy, 1=sell
    output reg  [2:0]               book_action,    // 0=add, 1=modify, 2=delete
    
    // Statistics
    output wire [31:0]              packets_processed,
    output wire [31:0]              parse_errors,
    output wire [15:0]              pipeline_depth
);

// Internal registers
reg [31:0] packet_counter;
reg [31:0] error_counter;
reg [DATA_WIDTH-1:0] data_buffer;
reg [7:0] current_msg_type;
reg [2:0] parse_state;
reg [15:0] msg_length;
reg [15:0] bytes_processed;

// Pipeline registers for latency optimization
reg [DATA_WIDTH-1:0] pipeline_stage1;
reg [DATA_WIDTH-1:0] pipeline_stage2;
reg [DATA_WIDTH-1:0] pipeline_stage3;
reg valid_stage1, valid_stage2, valid_stage3;

// ITCH message parsing state machine
localparam IDLE = 3'b000;
localparam HEADER = 3'b001;
localparam PAYLOAD = 3'b010;
localparam VALIDATE = 3'b011;
localparam OUTPUT = 3'b100;

// ITCH message types
localparam MSG_ADD_ORDER = 8'h41;           // 'A'
localparam MSG_EXECUTE_ORDER = 8'h45;       // 'E'
localparam MSG_CANCEL_ORDER = 8'h58;        // 'X'
localparam MSG_DELETE_ORDER = 8'h44;        // 'D'
localparam MSG_REPLACE_ORDER = 8'h55;       // 'U'

// Internal wires
wire [PRICE_WIDTH-1:0] extracted_price;
wire [VOLUME_WIDTH-1:0] extracted_volume;
wire [SYMBOL_WIDTH-1:0] extracted_symbol;
wire [31:0] extracted_order_ref;
wire parse_complete;
wire parse_error;

// Pipeline control
assign data_ready = (parse_state == IDLE) || (parse_state == OUTPUT);
assign pipeline_depth = 16'd3;  // 3-stage pipeline

// Statistics outputs
assign packets_processed = packet_counter;
assign parse_errors = error_counter;

// Clock domain logic
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset all registers
        packet_counter <= 32'b0;
        error_counter <= 32'b0;
        parse_state <= IDLE;
        msg_length <= 16'b0;
        bytes_processed <= 16'b0;
        tick_valid <= 1'b0;
        book_update_valid <= 1'b0;
        
        // Pipeline reset
        pipeline_stage1 <= {DATA_WIDTH{1'b0}};
        pipeline_stage2 <= {DATA_WIDTH{1'b0}};
        pipeline_stage3 <= {DATA_WIDTH{1'b0}};
        valid_stage1 <= 1'b0;
        valid_stage2 <= 1'b0;
        valid_stage3 <= 1'b0;
        
    end else begin
        // Pipeline advancement
        pipeline_stage1 <= data_in;
        pipeline_stage2 <= pipeline_stage1;
        pipeline_stage3 <= pipeline_stage2;
        valid_stage1 <= data_valid;
        valid_stage2 <= valid_stage1;
        valid_stage3 <= valid_stage2;
        
        // Main parsing state machine
        case (parse_state)
            IDLE: begin
                tick_valid <= 1'b0;
                book_update_valid <= 1'b0;
                
                if (data_valid) begin
                    data_buffer <= data_in;
                    current_msg_type <= data_type;
                    parse_state <= HEADER;
                    bytes_processed <= 16'b1;
                end
            end
            
            HEADER: begin
                if (data_valid) begin
                    // Extract message length from header
                    msg_length <= data_in[15:0];
                    bytes_processed <= bytes_processed + 1;
                    parse_state <= PAYLOAD;
                end
            end
            
            PAYLOAD: begin
                if (data_valid) begin
                    bytes_processed <= bytes_processed + 1;
                    
                    // Parse based on message type
                    case (current_msg_type)
                        MSG_ADD_ORDER: begin
                            // Parse Add Order message
                            if (bytes_processed >= msg_length) begin
                                parse_state <= VALIDATE;
                            end
                        end
                        
                        MSG_EXECUTE_ORDER: begin
                            // Parse Execute Order message
                            if (bytes_processed >= msg_length) begin
                                parse_state <= VALIDATE;
                            end
                        end
                        
                        default: begin
                            // Unknown message type
                            error_counter <= error_counter + 1;
                            parse_state <= IDLE;
                        end
                    endcase
                end
            end
            
            VALIDATE: begin
                // Validate parsed data
                if (parse_error) begin
                    error_counter <= error_counter + 1;
                    parse_state <= IDLE;
                end else begin
                    parse_state <= OUTPUT;
                end
            end
            
            OUTPUT: begin
                // Output parsed data
                packet_counter <= packet_counter + 1;
                
                // Generate tick output
                tick_valid <= 1'b1;
                symbol <= extracted_symbol;
                price <= extracted_price;
                volume <= extracted_volume;
                timestamp <= $time;
                
                // Generate order book update
                book_update_valid <= 1'b1;
                book_symbol <= extracted_symbol;
                book_price <= extracted_price;
                book_volume <= extracted_volume;
                book_side <= (current_msg_type == MSG_ADD_ORDER) ? 1'b0 : 1'b1;
                book_action <= (current_msg_type == MSG_ADD_ORDER) ? 3'b000 : 
                              (current_msg_type == MSG_EXECUTE_ORDER) ? 3'b001 : 3'b010;
                
                parse_state <= IDLE;
            end
            
            default: begin
                parse_state <= IDLE;
            end
        endcase
    end
end

// Data extraction logic (combinational)
assign extracted_price = pipeline_stage3[31:0];
assign extracted_volume = pipeline_stage2[31:0];
assign extracted_symbol = pipeline_stage1[31:0];
assign extracted_order_ref = pipeline_stage3[63:32];

// Parse completion detection
assign parse_complete = (parse_state == VALIDATE) && !parse_error;

// Error detection logic
assign parse_error = (msg_length == 16'b0) || (msg_length > 16'd512) || 
                    (bytes_processed > msg_length + 16'd10);

// Bid/Ask calculation (simplified)
always @(*) begin
    case (current_msg_type)
        MSG_ADD_ORDER: begin
            if (book_side == 1'b0) begin  // Buy order
                bid = extracted_price;
                ask = extracted_price + 32'd1;  // Simplified spread
            end else begin  // Sell order
                ask = extracted_price;
                bid = extracted_price - 32'd1;
            end
        end
        
        default: begin
            bid = 32'b0;
            ask = 32'b0;
        end
    endcase
end

endmodule
