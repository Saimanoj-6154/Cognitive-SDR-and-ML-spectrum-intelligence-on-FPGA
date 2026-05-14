module spectrum_sensing #(
    parameter THRESHOLD = 32'd1000
)(
    input wire clk,
    input wire rst,
    input wire [31:0] power_in,
    input wire valid_in,

    output reg occupied,
    output reg valid_out
);

always @(posedge clk) begin
    if (rst) begin
        occupied <= 0;
        valid_out <= 0;
    end else if (valid_in) begin
        occupied <= (power_in > THRESHOLD);
        valid_out <= 1;
    end
end

endmodule
