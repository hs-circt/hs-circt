// circt-opt %s --memory-mapping
// RUN: circt-opt %s --memory-mapping | FileCheck %s

module{
    // CHECK-LABEL: @DCacheDataArray
    // CHECK: %RAMB36E2_1.DOADO = hw.instance "RAMB36E2_1" @RAMB36E2_0(CLKARDCLK: %clock: !seq.clock, ADDRARDADDR: %0: i9, WEA: %io_req_bits_write: i1, ENA: %1: i1, DIADI: %io_req_bits_wdata: i32, REGCEA: %true_0: i1, RSTRAMA: %false: i1, RSTREGA: %false: i1, MASK: %false_1: i1)
  hw.module private @DCacheDataArray(in %clock : !seq.clock, in %io_req_valid : i1, in %io_req_bits_addr : i12, in %io_req_bits_write : i1, in %io_req_bits_wdata : i32, in %io_req_bits_eccMask : i8, out io_resp_0 : i32) {
    %true = hw.constant true
    %0 = comb.extract %io_req_bits_addr from 3 {sv.namehint = "addr"} : (i12) -> i9
    %data_arrays_0 = seq.firmem 1, 1, undefined, port_order : <512 x 32>
    %1 = seq.firmem.read_write_port %data_arrays_0[%0] = %io_req_bits_wdata if %io_req_bits_write, clock %clock enable %2 : <512 x 32>
    %2 = comb.or bin %5, %3 : i1
    %3 = comb.and bin %io_req_valid, %io_req_bits_write {sv.namehint = "data_arrays_0_rdata_MPORT_en"} : i1
    %4 = comb.xor bin %io_req_bits_write, %true {sv.namehint = "_rdata_data_T"} : i1
    %5 = comb.and bin %io_req_valid, %4 {sv.namehint = "data_arrays_0_rdata_data_en"} : i1
    hw.output %1 : i32
  }
  // CHECK-LABEL: @DCacheDataArrayE
  // CHECK: %RAMB36E2_3.DOADO = hw.instance "RAMB36E2_3" @RAMB36E2_2(CLKARDCLK: %clock: !seq.clock, ADDRARDADDR: %0: i9, WEA: %io_req_bits_write: i1, ENA: %1: i1, DIADI: %io_req_bits_wdata: i64, REGCEA: %true_0: i1, RSTRAMA: %false: i1, RSTREGA: %false: i1, MASK: %io_req_bits_eccMask: i8)
  hw.module private @DCacheDataArrayE(in %clock : !seq.clock, in %io_req_valid : i1, in %io_req_bits_addr : i12, in %io_req_bits_write : i1, in %io_req_bits_wdata : i64, in %io_req_bits_eccMask : i8, out io_resp_0 : i64) {
    %true = hw.constant true
    %0 = comb.extract %io_req_bits_addr from 3 {sv.namehint = "addr"} : (i12) -> i9
    %data_arrays_0 = seq.firmem 1, 1, undefined, port_order : <512 x 64, mask 8>
    %1 = seq.firmem.read_write_port %data_arrays_0[%0] = %io_req_bits_wdata if %io_req_bits_write, clock %clock enable %2 mask %io_req_bits_eccMask : <512 x 64, mask 8>, i8
    %2 = comb.or bin %5, %3 : i1
    %3 = comb.and bin %io_req_valid, %io_req_bits_write {sv.namehint = "data_arrays_0_rdata_MPORT_en"} : i1
    %4 = comb.xor bin %io_req_bits_write, %true {sv.namehint = "_rdata_data_T"} : i1
    %5 = comb.and bin %io_req_valid, %4 {sv.namehint = "data_arrays_0_rdata_data_en"} : i1
    hw.output %1 : i64
  }
  // CHECK-LABEL: @Queue8_TLBroadcastData
  // CHECK: %0 = comb.extract %61 from 0 : (i72) -> i2
  // CHECK: %{{.*}} = hw.instance "RAM32M_{{.*}}" @RAM32M_4
  hw.module private @Queue8_TLBroadcastData(in %clock : !seq.clock, in %reset : i1, out io_enq_ready : i1, in %io_enq_valid : i1, in %io_enq_bits_mask : i8, in %io_enq_bits_data : i64, in %io_deq_ready : i1, out io_deq_valid : i1, out io_deq_bits_mask : i8, out io_deq_bits_data : i64) {
    %c1_i3 = hw.constant 1 : i3
    %false = hw.constant false
    %c0_i3 = hw.constant 0 : i3
    %true = hw.constant true
    %ram = seq.firmem 0, 1, undefined, port_order : <8 x 72>
    seq.firmem.write_port %ram[%enq_ptr_value] = %1, clock %clock enable %8 : <8 x 72>
    %0 = seq.firmem.read_port %ram[%deq_ptr_value], clock %clock : <8 x 72>
    %1 = comb.concat %io_enq_bits_mask, %io_enq_bits_data : i8, i64
    %2 = comb.extract %0 from 64 {sv.namehint = "ram_io_deq_bits_MPORT_data_mask"} : (i72) -> i8
    %3 = comb.extract %0 from 0 {sv.namehint = "ram_io_deq_bits_MPORT_data_data"} : (i72) -> i64
    %enq_ptr_value = seq.firreg %11 clock %clock reset sync %reset, %c0_i3 {firrtl.random_init_start = 0 : ui64, sv.namehint = "enq_ptr_value"} : i3
    %deq_ptr_value = seq.firreg %13 clock %clock reset sync %reset, %c0_i3 {firrtl.random_init_start = 3 : ui64, sv.namehint = "deq_ptr_value"} : i3
    %maybe_full = seq.firreg %15 clock %clock reset sync %reset, %false {firrtl.random_init_start = 6 : ui64} : i1
    %4 = comb.icmp bin eq %enq_ptr_value, %deq_ptr_value {sv.namehint = "ptr_match"} : i3
    %5 = comb.xor bin %maybe_full, %true {sv.namehint = "_empty_T"} : i1
    %6 = comb.and bin %4, %5 {sv.namehint = "empty"} : i1
    %7 = comb.and bin %4, %maybe_full {sv.namehint = "full"} : i1
    %8 = comb.and bin %17, %io_enq_valid {sv.namehint = "do_enq"} : i1
    %9 = comb.and bin %io_deq_ready, %16 {sv.namehint = "do_deq"} : i1
    %10 = comb.add bin %enq_ptr_value, %c1_i3 {sv.namehint = "_value_T"} : i3
    %11 = comb.mux bin %8, %10, %enq_ptr_value : i3
    %12 = comb.add bin %deq_ptr_value, %c1_i3 {sv.namehint = "_value_T_2"} : i3
    %13 = comb.mux bin %9, %12, %deq_ptr_value : i3
    %14 = comb.icmp bin eq %8, %9 : i1
    %15 = comb.mux bin %14, %maybe_full, %8 : i1
    %16 = comb.xor bin %6, %true {sv.namehint = "io_deq_valid"} : i1
    %17 = comb.xor bin %7, %true {sv.namehint = "io_enq_ready"} : i1
    hw.output %17, %16, %2, %3 : i1, i1, i8, i64
  }
  // CHECK-LABEL: @Queue2_TLBundleE
  // CHECK: %0 = comb.extract %io_enq_bits_sink from 0 : (i2) -> i1
  // CHECK: %RAM32X1D_18.DPO, %RAM32X1D_18.SPO = hw.instance "RAM32X1D_18" @RAM32X1D_17
  hw.module private @Queue2_TLBundleE(in %clock : !seq.clock, in %reset : i1, out io_enq_ready : i1, in %io_enq_valid : i1, in %io_enq_bits_sink : i2, out io_deq_valid : i1, out io_deq_bits_sink : i2) {
    %false = hw.constant false
    %true = hw.constant true
    %ram_sink = seq.firmem 0, 1, undefined, port_order : <2 x 2>
    seq.firmem.write_port %ram_sink[%wrap] = %io_enq_bits_sink, clock %clock enable %5 : <2 x 2>
    %0 = seq.firmem.read_port %ram_sink[%wrap_1], clock %clock {sv.namehint = "ram_io_deq_bits_MPORT_data_sink"} : <2 x 2>
    %wrap = seq.firreg %7 clock %clock reset sync %reset, %false {firrtl.random_init_start = 0 : ui64, sv.namehint = "wrap"} : i1
    %wrap_1 = seq.firreg %9 clock %clock reset sync %reset, %false {firrtl.random_init_start = 1 : ui64, sv.namehint = "wrap_1"} : i1
    %maybe_full = seq.firreg %11 clock %clock reset sync %reset, %false {firrtl.random_init_start = 2 : ui64} : i1
    %1 = comb.icmp bin eq %wrap, %wrap_1 {sv.namehint = "ptr_match"} : i1
    %2 = comb.xor bin %maybe_full, %true {sv.namehint = "_empty_T"} : i1
    %3 = comb.and bin %1, %2 {sv.namehint = "empty"} : i1
    %4 = comb.and bin %1, %maybe_full {sv.namehint = "full"} : i1
    %5 = comb.and bin %13, %io_enq_valid {sv.namehint = "do_enq"} : i1
    %6 = comb.add bin %wrap, %true {sv.namehint = "_value_T"} : i1
    %7 = comb.mux bin %5, %6, %wrap : i1
    %8 = comb.add bin %wrap_1, %true {sv.namehint = "_value_T_2"} : i1
    %9 = comb.mux bin %3, %wrap_1, %8 : i1
    %10 = comb.icmp bin eq %5, %12 : i1
    %11 = comb.mux bin %10, %maybe_full, %5 : i1
    %12 = comb.xor bin %3, %true {sv.namehint = "do_deq"} : i1
    %13 = comb.xor bin %4, %true {sv.namehint = "io_enq_ready"} : i1
    hw.output %13, %12, %0 : i1, i1, i2
  }
  // CHECK-LABEL: @Queue2_AXI4BundleW
  // CHECK: %0 = comb.extract %64 from 0 : (i73) -> i2
  // CHECK: %RAM32M_20.DOA, %RAM32M_20.DOB, %RAM32M_20.DOC, %RAM32M_20.DOD = hw.instance "RAM32M_20" @RAM32M_4
  hw.module private @Queue2_AXI4BundleW(in %clock : !seq.clock, in %reset : i1, out io_enq_ready : i1, in %io_enq_valid : i1, in %io_enq_bits_data : i64, in %io_enq_bits_strb : i8, in %io_enq_bits_last : i1, in %io_deq_ready : i1, out io_deq_valid : i1, out io_deq_bits_data : i64, out io_deq_bits_strb : i8, out io_deq_bits_last : i1) {
    %false = hw.constant false
    %true = hw.constant true
    %ram = seq.firmem 0, 1, undefined, port_order : <2 x 73>
    seq.firmem.write_port %ram[%wrap] = %1, clock %clock enable %9 : <2 x 73>
    %0 = seq.firmem.read_port %ram[%wrap_1], clock %clock : <2 x 73>
    %1 = comb.concat %io_enq_bits_data, %io_enq_bits_strb, %io_enq_bits_last : i64, i8, i1
    %2 = comb.extract %0 from 9 {sv.namehint = "ram_io_deq_bits_MPORT_data_data"} : (i73) -> i64
    %3 = comb.extract %0 from 1 {sv.namehint = "ram_io_deq_bits_MPORT_data_strb"} : (i73) -> i8
    %4 = comb.extract %0 from 0 {sv.namehint = "ram_io_deq_bits_MPORT_data_last"} : (i73) -> i1
    %wrap = seq.firreg %12 clock %clock reset sync %reset, %false {firrtl.random_init_start = 0 : ui64, sv.namehint = "wrap"} : i1
    %wrap_1 = seq.firreg %14 clock %clock reset sync %reset, %false {firrtl.random_init_start = 1 : ui64, sv.namehint = "wrap_1"} : i1
    %maybe_full = seq.firreg %16 clock %clock reset sync %reset, %false {firrtl.random_init_start = 2 : ui64} : i1
    %5 = comb.icmp bin eq %wrap, %wrap_1 {sv.namehint = "ptr_match"} : i1
    %6 = comb.xor bin %maybe_full, %true {sv.namehint = "_empty_T"} : i1
    %7 = comb.and bin %5, %6 {sv.namehint = "empty"} : i1
    %8 = comb.and bin %5, %maybe_full {sv.namehint = "full"} : i1
    %9 = comb.and bin %18, %io_enq_valid {sv.namehint = "do_enq"} : i1
    %10 = comb.and bin %io_deq_ready, %17 {sv.namehint = "do_deq"} : i1
    %11 = comb.add bin %wrap, %true {sv.namehint = "_value_T"} : i1
    %12 = comb.mux bin %9, %11, %wrap : i1
    %13 = comb.add bin %wrap_1, %true {sv.namehint = "_value_T_2"} : i1
    %14 = comb.mux bin %10, %13, %wrap_1 : i1
    %15 = comb.icmp bin eq %9, %10 : i1
    %16 = comb.mux bin %15, %maybe_full, %9 : i1
    %17 = comb.xor bin %7, %true {sv.namehint = "io_deq_valid"} : i1
    %18 = comb.xor bin %8, %true {sv.namehint = "io_enq_ready"} : i1
    hw.output %18, %17, %2, %3, %4 : i1, i1, i64, i8, i1
  }
}
  