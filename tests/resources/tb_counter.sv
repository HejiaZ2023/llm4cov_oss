module tb;
  logic clk, rstn;
  logic [3:0] q;

  counter dut (.clk(clk), .rstn(rstn), .q(q));

  // Simple functional coverage
  covergroup cg @(posedge clk);
    coverpoint q {
      bins low = {0,1,2};
      bins mid = {[3:7]};
      bins high = {[8:15]};
    }
  endgroup
  cg cov = new();

  // Clock
  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    rstn = 0;
    repeat (2) @(posedge clk);
    rstn = 1;
    repeat (20) @(posedge clk);
    $finish;
  end
endmodule
