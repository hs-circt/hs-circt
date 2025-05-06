module FDRE #(
    parameter [0:0] INIT = 1'b0
)(
    input  wire C,
    input  wire D,
    input  wire CE,
    input  wire R,
    output reg  Q
);
  always @(posedge C) begin
      if (R) begin
          Q <= 1'b0;
      end else if (CE) begin
          Q <= D;
      end
  end
  initial begin
      Q = INIT;
  end
endmodule

module FDSE #(
    parameter [0:0] INIT = 1'b0
)(
    input  wire C,
    input  wire D,
    input  wire CE,
    input  wire S,
    output reg  Q
);
  always @(posedge C) begin
      if (S) begin
          Q <= 1'b1;
      end else if (CE) begin
          Q <= D;
      end
  end
  initial begin
      Q = INIT;
  end
endmodule


module LUT2 #(
    parameter [3:0] INIT = 0
)(
    output O,
    input I0, I1
);
  wire [1:0] s1 = I1 ? INIT[3:2] : INIT[1:0];
  assign O = I0 ? s1[1] : s1[0];
endmodule
