#include "Vmain_tb.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <stdio.h>

// 定义仿真时间函数
double sc_time_stamp() {
    return 0;
}

int main(int argc, char** argv) {
    // 启用Verilog的$display输出重定向
    Verilated::commandArgs(argc, argv);
    
    // 创建仿真上下文
    Vmain_tb* tb = new Vmain_tb;
    
    // 重定向Verilog标准输出
    Verilated::fatalOnVpiError(false);  // 不在VPI错误时终止
    
    // 创建波形跟踪器
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    tb->trace(tfp, 99);
    tfp->open("waveform.vcd");
    
    // 仿真周期
    vluint64_t main_time = 0;
    int sim_cycles = 2000;  // 增加最大仿真周期数以支持两个测试模块
    bool clk_val = 0;
    
    // 初始化 - 使用main_tb__DOT__前缀访问变量
    tb->main_tb__DOT__clock = 0;
    tb->main_tb__DOT__reset = 1;
    tb->main_tb__DOT__io_enq_valid = 0;
    tb->main_tb__DOT__io_enq_bits_data = 0;
    tb->main_tb__DOT__io_enq_bits_strb = 0;
    tb->main_tb__DOT__io_enq_bits_last = 0;
    tb->main_tb__DOT__io_deq_ready = 0;
    tb->main_tb__DOT__test_state = 0;
    tb->main_tb__DOT__test_counter = 0;
    tb->main_tb__DOT__test_mode = 0;     // 开始时运行Queue测试
    
    // 初始化Cache测试信号
    tb->main_tb__DOT__io_req_valid = 0;
    tb->main_tb__DOT__io_req_bits_addr = 0;
    tb->main_tb__DOT__io_req_bits_write = 0;
    tb->main_tb__DOT__io_req_bits_wdata = 0;
    tb->main_tb__DOT__io_req_bits_eccMask = 0;
    
    // 仿真主循环
    while (!Verilated::gotFinish() && main_time < sim_cycles) {
        // 评估模型（先评估一次）
        tb->eval();
        
        // 时钟切换
        clk_val = !clk_val;
        tb->main_tb__DOT__clock = clk_val;  // 使用正确的变量名访问
        
        // 评估模型（切换时钟后再评估一次）
        tb->eval();
        
        // 当时钟上升沿且已经运行一段时间后，停止复位
        if (clk_val && main_time == 10) {
            tb->main_tb__DOT__reset = 0;
            printf("C++: Reset completed at time %lu\n", main_time);
        }
        
        // 在时钟上升沿进行测试操作
        if (clk_val && (main_time % 20 == 0)) {
            // 显示C++侧看到的状态信息
            if (tb->main_tb__DOT__test_mode == 0) {
                // Queue测试状态信息
                printf("C++: Time: %lu, Mode: Queue, State: %d, Counter: %d\n", 
                       main_time, 
                       tb->main_tb__DOT__test_state, 
                       tb->main_tb__DOT__test_counter);
            } else {
                // Cache测试状态信息
                printf("C++: Time: %lu, Mode: Cache, State: %d, Counter: %d\n", 
                       main_time, 
                       tb->main_tb__DOT__test_state, 
                       tb->main_tb__DOT__test_counter);
            }
        }
        
        // 记录波形
        tfp->dump(main_time);
        
        // 仿真时间前进
        main_time++;
    }
    
    // 清理
    tfp->close();
    delete tfp;
    delete tb;
    
    printf("C++: Simulation completed at time %lu\n", main_time);
    return 0;
} 