module main_tb;
  // Queue2_AXI4BundleW信号定义
  reg         clock;
  reg         reset;
  reg         io_enq_valid;
  reg  [63:0] io_enq_bits_data;
  reg  [7:0]  io_enq_bits_strb;
  reg         io_enq_bits_last;
  reg         io_deq_ready;

  wire        io_enq_ready;
  wire        io_deq_valid;
  wire [63:0] io_deq_bits_data;
  wire [7:0]  io_deq_bits_strb;
  wire        io_deq_bits_last;

  // DCacheDataArrayL信号定义
  reg         io_req_valid;
  reg  [11:0] io_req_bits_addr;
  reg         io_req_bits_write;
  reg  [31:0] io_req_bits_wdata;
  reg  [7:0]  io_req_bits_eccMask;
  wire [31:0] io_resp_0;

  // Queue2_AXI4BundleW64的信号（共享现有信号，因为接口相同）
  wire        io_enq_ready_64;
  wire        io_deq_valid_64;
  wire [63:0] io_deq_bits_data_64;
  wire [7:0]  io_deq_bits_strb_64;
  wire        io_deq_bits_last_64;

  // 测试状态机状态定义
  reg [4:0] test_state;  // 增加到5位以支持状态16
  integer   test_counter;
  integer   main_time = 0;
  
  // 测试模式（0 = Queue测试, 1 = Cache测试, 2 = Queue64测试）
  reg [1:0] test_mode;

  // 实例化Queue2_AXI4BundleW模块
  Queue2_AXI4BundleW queue_dut (
    .clock             (clock),
    .reset             (reset),
    .io_enq_ready      (io_enq_ready),
    .io_enq_valid      (io_enq_valid),
    .io_enq_bits_data  (io_enq_bits_data),
    .io_enq_bits_strb  (io_enq_bits_strb),
    .io_enq_bits_last  (io_enq_bits_last),
    .io_deq_ready      (io_deq_ready),
    .io_deq_valid      (io_deq_valid),
    .io_deq_bits_data  (io_deq_bits_data),
    .io_deq_bits_strb  (io_deq_bits_strb),
    .io_deq_bits_last  (io_deq_bits_last)
  );
  
  // 实例化DCacheDataArrayL模块
  DCacheDataArrayL cache_dut (
    .clock             (clock),
    .io_req_valid      (io_req_valid),
    .io_req_bits_addr  (io_req_bits_addr),
    .io_req_bits_write (io_req_bits_write),
    .io_req_bits_wdata (io_req_bits_wdata),
    .io_req_bits_eccMask (io_req_bits_eccMask),
    .io_resp_0         (io_resp_0)
  );

  // 实例化Queue2_AXI4BundleW64模块
  Queue2_AXI4BundleW64 queue64_dut (
    .clock             (clock),
    .reset             (reset),
    .io_enq_ready      (io_enq_ready_64),
    .io_enq_valid      (io_enq_valid),
    .io_enq_bits_data  (io_enq_bits_data),
    .io_enq_bits_strb  (io_enq_bits_strb),
    .io_enq_bits_last  (io_enq_bits_last),
    .io_deq_ready      (io_deq_ready),
    .io_deq_valid      (io_deq_valid_64),
    .io_deq_bits_data  (io_deq_bits_data_64),
    .io_deq_bits_strb  (io_deq_bits_strb_64),
    .io_deq_bits_last  (io_deq_bits_last_64)
  );

  // 初始化
  initial begin
    // 共享信号
    clock = 0;
    reset = 1;
    test_state = 0;
    test_counter = 0;
    test_mode = 0; // 先测试Queue
    
    // Queue信号初始化
    io_enq_valid = 0;
    io_enq_bits_data = 0;
    io_enq_bits_strb = 0;
    io_enq_bits_last = 0;
    io_deq_ready = 0;
    
    // Cache信号初始化
    io_req_valid = 0;
    io_req_bits_addr = 0;
    io_req_bits_write = 0;
    io_req_bits_wdata = 0;
    io_req_bits_eccMask = 0;
  end

  // Clock tracking for Verilator, controlled by C++ driver
  always @(posedge clock) begin
    main_time <= main_time + 1;
  end

  // 测试状态机
  always @(posedge clock) begin
    if (test_mode == 0) begin
      // Queue2_AXI4BundleW测试
      case (test_state)
        // 重置阶段
        0: begin
          if (test_counter < 2) begin
            test_counter <= test_counter + 1;
          end else begin
            reset <= 0;
            test_state <= 1;
            test_counter <= 0;
            $display("Reset completed, starting Queue test...");
          end
        end

        // 测试1：初始状态验证
        1: begin
          if (io_enq_ready !== 1'b1 || io_deq_valid !== 1'b0) begin
            $display("Error: After reset, enq_ready should be 1, deq_valid should be 0");
            $finish;
          end
          test_state <= 2;
        end

        // 测试2：单次入队准备
        2: begin
          io_enq_valid <= 1;
          io_enq_bits_data <= 64'hA1B2C3D4E5F6_7F;
          io_enq_bits_strb <= 8'hFF;
          io_enq_bits_last <= 1;
          test_state <= 3;
        end

        // 测试2：单次入队验证
        3: begin
          if (!io_enq_ready) begin
            $display("Error: Queue should keep enq_ready=1 when not full");
            $finish;
          end
          test_state <= 4;
        end

        // 测试2：单次出队准备
        4: begin
          io_enq_valid <= 0;
          io_deq_ready <= 1;
          test_state <= 5;
        end

        // 测试2：单次出队验证
        5: begin
          if (io_deq_valid !== 1'b1 || 
              io_deq_bits_data !== 64'hA1B2C3D4E5F6_7F ||
              io_deq_bits_strb !== 8'hFF ||
              io_deq_bits_last !== 1'b1) begin
            $display("Error: Dequeued data mismatch");
            $display("Expected: data=0x%h, strb=0x%h, last=%b", 
                     64'hA1B2C3D4E5F6_7F, 8'hFF, 1'b1);
            $display("Actual: data=0x%h, strb=0x%h, last=%b",
                     io_deq_bits_data, io_deq_bits_strb, io_deq_bits_last);
            $finish;
          end
          test_state <= 6;
          io_deq_ready <= 0;
        end

        // 测试3：队列满状态准备
        6: begin
          io_enq_valid <= 1;
          io_enq_bits_data <= 64'h1122_3344_5566_7788;
          io_enq_bits_strb <= 8'hAA;
          io_enq_bits_last <= 0;
          test_state <= 7;
        end

        // 测试3：第一次入队后检查
        7: begin
          if (!io_enq_ready) begin
            $display("Error: Should have space after first enqueue");
            $finish;
          end
          io_enq_bits_data <= 64'h99AA_BBCC_DDEE_FF00;
          io_enq_bits_strb <= 8'h55;
          io_enq_bits_last <= 1;
          test_state <= 8;
        end

        // 测试3：第二次入队后检查（队列应满）
        8: begin
          if (io_enq_ready !== 1'b0) begin
            $display("Error: Queue should be full with enq_ready=0");
            $finish;
          end
          test_state <= 9;
        end

        // 测试4：同时读写测试
        9: begin
          io_deq_ready <= 1;
          io_enq_bits_data <= 64'h1234_5678_9ABC_DEF0;
          test_state <= 10;
        end

        // 测试4：验证流水线行为
        10: begin
          if (test_counter < 2) begin
              // 等待两个时钟周期让队列状态完全更新
              test_counter <= test_counter + 1;
              $display("Waiting cycle %0d for queue state to update", test_counter);
          end else if (io_enq_ready !== 1'b1) begin
              $display("Error: Should recover space during simultaneous read/write after 2 cycles");
              test_state <= 11;  // 即使失败也继续到下一个测试
              test_counter <= 0;
              $finish;
          end else begin
              $display("Success: Queue recovered space correctly, io_enq_ready = %b", io_enq_ready);
              test_state <= 11;
              test_counter <= 0;
          end
        end

        // 测试5：空队列准备
        11: begin
          io_enq_valid <= 0;  // 确保没有新数据入队
          io_deq_ready <= 1;  // 启用出队
          
          if (test_counter < 5) begin
            test_counter <= test_counter + 1;
            $display("Empty test - waiting cycle %0d, io_deq_valid = %b", test_counter, io_deq_valid);
          end else begin
            test_state <= 12;
            test_counter <= 0;
          end
        end

        // 测试5：验证队列为空
        12: begin
          if (io_deq_valid !== 1'b0) begin
            $display("Error: deq_valid should be 0 when queue is empty");
            test_state <= 13;
            $finish;
          end else begin
            $display("Queue is empty as expected");
            test_state <= 13;
          end
        end

        // Queue测试完成，切换到Cache测试
        13: begin
          $display("Queue tests completed! Switching to Cache tests...");
          test_mode <= 1;  // 切换到Cache测试
          test_state <= 0; // 重置测试状态
          test_counter <= 0;
        end
        
        default: begin
          test_state <= 0;
        end
      endcase
    end
    else if (test_mode == 1) begin
      // DCacheDataArrayL测试
      case (test_state)
        // 初始化阶段
        0: begin
          $display("Reset completed, starting Cache test...");
          io_req_valid <= 0;
          io_req_bits_addr <= 0;
          io_req_bits_write <= 0;
          io_req_bits_wdata <= 0;
          io_req_bits_eccMask <= 0;
          test_state <= 1;
        end

        // 测试1：基础写入准备
        1: begin
          $display("[TEST1] 基础写读测试");
          io_req_valid <= 1;
          io_req_bits_write <= 1;
          io_req_bits_addr <= 12'hAAA;
          io_req_bits_wdata <= 32'h12345678;
          io_req_bits_eccMask <= 8'hFF;
          test_state <= 2;
        end

        // 测试1：切换为读取模式
        2: begin
          io_req_bits_write <= 0;
          test_state <= 3;
        end

        // 测试1：验证读取数据
        3: begin
          if (io_resp_0 !== 32'h12345678) begin
            $display("Error: Read data 0x%h doesn't match written value 0x12345678", io_resp_0);
            $finish;
          end
          test_state <= 4;
        end

        // 测试2：边界地址写入准备
        4: begin
          $display("[TEST2] 边界地址测试");
          io_req_valid <= 1;
          io_req_bits_write <= 1;
          io_req_bits_addr <= 12'hFFF;
          io_req_bits_wdata <= 32'hDEADBEEF;
          test_state <= 5;
        end

        // 测试2：切换为读取模式
        5: begin
          io_req_bits_write <= 0;
          test_state <= 6;
        end

        // 测试2：验证边界地址读取
        6: begin
          if (io_resp_0 !== 32'hDEADBEEF) begin
            $display("Error: Boundary address 0xFFF read data 0x%h is incorrect", io_resp_0);
            $finish;
          end
          test_state <= 7;
        end

        // 测试3A：全1模式写入准备
        7: begin
          $display("[TEST3] 全位模式测试");
          io_req_valid <= 1;
          io_req_bits_write <= 1;
          io_req_bits_addr <= 12'h555;
          io_req_bits_wdata <= 32'hFFFFFFFF;
          test_state <= 8;
        end

        // 测试3A：切换为读取模式
        8: begin
          io_req_bits_write <= 0;
          test_state <= 9;
        end

        // 测试3A：验证全1模式读取
        9: begin
          if (io_resp_0 !== 32'hFFFFFFFF) begin
            $display("Error: All-ones pattern read error, got 0x%h", io_resp_0);
            $finish;
          end else begin
            $display("测试3A通过：地址0x555成功读取到0xFFFFFFFF");
          end
          
          // 准备全0模式写入到不同地址
          io_req_valid <= 1;
          io_req_bits_write <= 1;
          io_req_bits_addr <= 12'h666;  // 使用不同地址，避免覆盖0x555的全1数据
          io_req_bits_wdata <= 32'h0;
          test_state <= 10;
        end

        // 测试3B：切换为全0读取模式
        10: begin
          io_req_bits_write <= 0;
          test_state <= 11;
        end

        // 测试3B：验证全0模式读取
        11: begin
          if (io_resp_0 !== 32'h0) begin
            $display("Error: All-zeros pattern read error, got 0x%h", io_resp_0);
            $finish;
          end else begin
            $display("测试3B通过：地址0x666成功读取到0x0");
          end
          test_state <= 12;
        end

        // 测试4：读写冲突测试准备
        12: begin
          $display("[TEST4] 同时读写测试");
          io_req_valid <= 1;
          io_req_bits_write <= 1;
          io_req_bits_addr <= 12'h800;
          io_req_bits_wdata <= 32'hA5A5A5A5;
          test_state <= 13;
        end

        // 测试4：切换为读取模式
        13: begin
          io_req_bits_write <= 0;
          test_state <= 14;
        end

        // 测试4：验证读写冲突行为
        14: begin
          if (io_resp_0 !== 32'hA5A5A5A5) begin
            $display("Error: Read-write conflict produced unexpected data, got 0x%h", io_resp_0);
            $finish;
          end else begin
            $display("测试4通过：成功读取0xA5A5A5A5");
          end
          
          // 开始测试5：再次读取地址0x555的数据
          $display("[TEST5] 再次读取地址0x555的全1数据");
          test_state <= 15;
          test_counter <= 0;
        end

        // 测试5：准备读取之前写入的全1数据
        15: begin
          io_req_valid <= 1;  // 确保请求有效
          io_req_bits_write <= 0;  // 读取模式
          io_req_bits_addr <= 12'h555;  // 设置为之前写入全1的地址
          
          // 使用计数器等待两个周期让存储器响应
          if (test_counter < 2) begin
            test_counter <= test_counter + 1;
            $display("Waiting cycle %0d for memory to respond to address 0x555", test_counter);
          end else begin
            test_state <= 16;
            test_counter <= 0;
            $display("验证地址0x555的数据...");
          end
        end

        // 测试5：验证全1数据读取
        16: begin
          if (io_resp_0 !== 32'hFFFFFFFF) begin
            $display("Error: 地址0x555期望值0xFFFFFFFF，但得到了0x%h", io_resp_0);
            $finish;
          end else begin
            $display("测试5通过：地址0x555成功读取到0xFFFFFFFF");
          end
          
          test_state <= 17;
        end
        
        // 测试完成 - 切换到Queue64测试
        17: begin
          $display("All Cache tests passed! Switching to Queue64 tests...");
          test_mode <= 2;  // 切换到Queue64测试
          test_state <= 0; // 重置测试状态
          test_counter <= 0;
          reset <= 1;      // 确保Queue64从重置状态开始
        end
        
        default: begin
          test_state <= 0;
        end
      endcase
    end
    else if (test_mode == 2) begin
      // Queue2_AXI4BundleW64测试
      case (test_state)
        // 重置阶段
        0: begin
          if (test_counter < 2) begin
            test_counter <= test_counter + 1;
          end else begin
            reset <= 0;
            test_state <= 1;
            test_counter <= 0;
            $display("[Queue64] Reset completed, starting test...");
          end
        end

        // 测试1：初始状态验证
        1: begin
          if (io_enq_ready_64 !== 1'b1 || io_deq_valid_64 !== 1'b0) begin
            $display("[Queue64] Error: After reset, enq_ready should be 1, deq_valid should be 0");
            $finish;
          end else begin
            $display("[Queue64] Initial state check passed");
          end
          test_state <= 2;
        end

        // 测试2：单次入队准备
        2: begin
          io_enq_valid <= 1;
          io_enq_bits_data <= 64'hA1B2C3D4E5F67890;
          io_enq_bits_strb <= 8'hAA;
          io_enq_bits_last <= 1;
          test_state <= 3;
          $display("[Queue64] Enqueueing data: 0x%h", 64'hA1B2C3D4E5F67890);
        end

        // 测试2：单次入队验证
        3: begin
          if (!io_enq_ready_64) begin
            $display("[Queue64] Error: Queue should keep enq_ready=1 when not full");
            $finish;
          end
          test_state <= 4;
        end

        // 测试2：单次出队准备
        4: begin
          io_enq_valid <= 0;
          io_deq_ready <= 1;
          test_state <= 5;
          $display("[Queue64] Dequeueing data...");
        end

        // 测试2：单次出队验证
        5: begin
          if (io_deq_valid_64 !== 1'b1 || 
              io_deq_bits_data_64 !== 64'hA1B2C3D4E5F67890 ||
              io_deq_bits_strb_64 !== 8'hAA ||
              io_deq_bits_last_64 !== 1'b1) begin
            $display("[Queue64] Error: Dequeued data mismatch");
            $display("[Queue64] Expected: data=0x%h, strb=0x%h, last=%b", 
                     64'hA1B2C3D4E5F67890, 8'hAA, 1'b1);
            $display("[Queue64] Actual: data=0x%h, strb=0x%h, last=%b",
                     io_deq_bits_data_64, io_deq_bits_strb_64, io_deq_bits_last_64);
            $finish;
          end else begin
            $display("[Queue64] Single enqueue/dequeue test passed");
          end
          test_state <= 6;
          io_deq_ready <= 0;
        end

        // 测试3：多数据入队测试 - 第一个数据
        6: begin
          io_enq_valid <= 1;
          io_enq_bits_data <= 64'h1122334455667788;
          io_enq_bits_strb <= 8'h55;
          io_enq_bits_last <= 0;
          test_state <= 7;
          $display("[Queue64] Multi-data test: Enqueueing first data");
        end

        // 测试3：多数据入队测试 - 第二个数据
        7: begin
          // 检查第一个数据是否成功入队
          if (!io_enq_ready_64) begin
            $display("[Queue64] Error: Queue should accept first data");
            $finish;
          end
          
          // 输入第二个数据
          io_enq_bits_data <= 64'h99AABBCCDDEEFF00;
          io_enq_bits_strb <= 8'h33;
          io_enq_bits_last <= 1;
          test_state <= 8;
          test_counter <= 0;
          $display("[Queue64] Multi-data test: Enqueueing second data");
        end

        // 测试3：检查第二个数据是否成功入队
        8: begin
          if (!io_enq_ready_64) begin
            $display("[Queue64] Error: Queue should accept second data");
            $finish;
          end
          $display("[Queue64] Queue has accepted both data items");
          
          // 注意：由于队列容量为64项，我们不测试满状态
          // 转到下一个测试：出队测试
          io_enq_valid <= 0;
          io_deq_ready <= 1;
          test_state <= 9;
        end

        // 测试4：出队验证 - 第一个数据
        9: begin
          if (!io_deq_valid_64) begin
            $display("[Queue64] Error: Queue should have valid data");
            $finish;
          end
          
          if (io_deq_bits_data_64 !== 64'h1122334455667788 || io_deq_bits_strb_64 !== 8'h55 || io_deq_bits_last_64 !== 0) begin
            $display("[Queue64] Error: First dequeued data mismatch");
            $display("[Queue64] Expected: data=0x1122334455667788, strb=0x55, last=0");
            $display("[Queue64] Got: data=0x%h, strb=0x%h, last=%b", io_deq_bits_data_64, io_deq_bits_strb_64, io_deq_bits_last_64);
            $finish;
          end
          
          $display("[Queue64] First data dequeued successfully");
          test_state <= 10;
        end

        // 测试4：出队验证 - 第二个数据
        10: begin
          if (!io_deq_valid_64) begin
            $display("[Queue64] Error: Queue should have valid data");
            $finish;
          end
          
          if (io_deq_bits_data_64 !== 64'h99AABBCCDDEEFF00 || io_deq_bits_strb_64 !== 8'h33 || io_deq_bits_last_64 !== 1) begin
            $display("[Queue64] Error: Second dequeued data mismatch");
            $display("[Queue64] Expected: data=0x99AABBCCDDEEFF00, strb=0x33, last=1");
            $display("[Queue64] Got: data=0x%h, strb=0x%h, last=%b", io_deq_bits_data_64, io_deq_bits_strb_64, io_deq_bits_last_64);
            $finish;
          end
          
          $display("[Queue64] Second data dequeued successfully");
          test_state <= 11;
        end

        // 测试4：检查队列为空
        11: begin
          io_deq_ready <= 0;
          if (io_deq_valid_64) begin
            $display("[Queue64] Error: Queue should be empty with deq_valid=0");
            $finish;
          end
          
          $display("[Queue64] Queue empty check passed");
          $display("[Queue64] All tests passed!");
          
          // 完成所有测试，结束流程
          $display("All tests completed successfully!");
          $finish;
        end
      endcase
    end
  end

  // 波形记录
  initial begin
    $dumpfile("waveform.vcd");
    $dumpvars(0, main_tb);
  end
endmodule