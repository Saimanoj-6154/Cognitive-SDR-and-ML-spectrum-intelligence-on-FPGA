`timescale 1ns / 1ps

module fft_tb;

reg clk;
reg rst;

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    rst = 1;
    #20;
    rst = 0;
end

endmodule
