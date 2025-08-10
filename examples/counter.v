// 8-bit Counter with Reset
module counter(
    input clk,
    input reset,
    input enable,
    output reg [7:0] count
);
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 8'b0;
        end else if (enable) begin
            count <= count + 1;
        end
    end
    
    // Expected properties:
    // 1. count == 0 after reset
    // 2. count increments when enabled
    // 3. count stays same when not enabled
    // 4. count wraps around at 255
endmodule