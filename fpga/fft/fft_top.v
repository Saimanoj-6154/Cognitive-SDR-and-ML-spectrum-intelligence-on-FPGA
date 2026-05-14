module fft_top (
    input wire clk,
    input wire rst,
    input wire signed [15:0] sample_real,
    input wire signed [15:0] sample_imag,
    input wire valid_in,

    output wire signed [31:0] fft_real,
    output wire signed [31:0] fft_imag,
    output wire valid_out
);

    // Xilinx FFT IP integration placeholder

endmodule
