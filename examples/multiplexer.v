// 4-to-1 Multiplexer
module multiplexer(
    input [1:0] sel,
    input [7:0] data0,
    input [7:0] data1, 
    input [7:0] data2,
    input [7:0] data3,
    output [7:0] out
);
    reg [7:0] out_reg;
    
    always @(*) begin
        case (sel)
            2'b00: out_reg = data0;
            2'b01: out_reg = data1;
            2'b10: out_reg = data2;
            2'b11: out_reg = data3;
            default: out_reg = 8'b0;
        endcase
    end
    
    assign out = out_reg;
    
    // Expected properties:
    // 1. out == data0 when sel == 00
    // 2. out == data1 when sel == 01
    // 3. out == data2 when sel == 10
    // 4. out == data3 when sel == 11
endmodule