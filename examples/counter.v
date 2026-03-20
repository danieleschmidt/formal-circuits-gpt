module counter (
    input clk,
    input rst,
    input enable,
    output reg [7:0] count
);
    always @(posedge clk) begin
        if (rst)
            count <= 8'b0;
        else if (enable)
            count <= count + 1;
    end
endmodule
