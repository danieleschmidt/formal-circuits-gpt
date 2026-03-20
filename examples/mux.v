module mux4 (
    input [1:0] sel,
    input [7:0] a, b, c, d,
    output reg [7:0] out
);
    always @(*) begin
        case (sel)
            2'b00: out = a;
            2'b01: out = b;
            2'b10: out = c;
            2'b11: out = d;
        endcase
    end
endmodule
