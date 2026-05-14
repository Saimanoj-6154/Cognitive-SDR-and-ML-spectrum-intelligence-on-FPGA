module channel_selector(
    input wire clk,
    input wire rst,
    input wire [5:0] occupancy,

    output reg [2:0] selected_channel
);

always @(posedge clk) begin
    if (rst)
        selected_channel <= 3'd0;

    else begin
        if (!occupancy[0]) selected_channel <= 3'd0;
        else if (!occupancy[1]) selected_channel <= 3'd1;
        else if (!occupancy[2]) selected_channel <= 3'd2;
        else if (!occupancy[3]) selected_channel <= 3'd3;
        else if (!occupancy[4]) selected_channel <= 3'd4;
        else selected_channel <= 3'd5;
    end
end

endmodule
