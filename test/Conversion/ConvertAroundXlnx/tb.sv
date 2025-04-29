// RUN:

module tb;
reg clock = 0;
always #5 clock = ~clock;

reg din = 1;

reg reset0 = 1, reset1 = 1;
initial begin
	#100;
	@(posedge clock);
	reset0 = 0;
	reset1 = 0;
	#20;
	@(posedge clock);
	din = 0;
	#20;
	@(posedge clock);
  $display("############################");
  $display("#           PASS           #");
  $display("############################");
	$finish;
end

wire out0, out1, out2;
wire out3, out4, out5;

GoldenTop inst_golden (.clock(clock), .clock_en(1), .reset0(reset0), .reset1(reset1), .in(din), .out0(out0), .out1(out1), .out2(out2));
CheckTop  inst_check  (.clock(clock), .clock_en(1), .reset0(reset0), .reset1(reset1), .in(din), .out0(out3), .out1(out4), .out2(out5));

always @(*) begin
  #1;
  $display("At time: %-d", $time);
	$display("[G] out0=%b, out1=%b, out2=%b", out0, out1, out2);
	$display("[C] out3=%b, out4=%b, out5=%b", out3, out4, out5);
  if ($time >= 100) begin
    assert (out0 == out3);
    assert (out1 == out4);
    assert (out2 == out5);
  end
end

endmodule
