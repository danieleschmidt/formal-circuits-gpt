// Simple 4-bit Adder Example for Formal Verification
module simple_adder(
    input [3:0] a,
    input [3:0] b,
    input cin,
    output [4:0] sum
);
    // Combinational adder with carry
    assign sum = a + b + cin;
    
    // Expected properties:
    // 1. sum >= a (overflow handling)
    // 2. sum >= b (overflow handling)
    // 3. sum == (a + b + cin) % 32 (modular arithmetic for 5-bit result)
endmodule