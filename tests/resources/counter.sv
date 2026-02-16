module counter #(parameter WIDTH=4) (
  input  logic        clk,
  input  logic        rstn,
  output logic [WIDTH-1:0] q
);
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn)
      q <= '0;
    else
      q <= q + 1;
  end
endmodule
