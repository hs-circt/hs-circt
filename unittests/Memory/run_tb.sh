rm -rf unittests/Memory/obj_dir

circt-opt test/Conversion/CoreMemoryMapping/test.mlir --memory-mapping --canonicalize -o unittests/Memory/output.mlir

firtool unittests/Memory/output.mlir -o unittests/Memory/output.v

# 先进入unittests/Memory目录运行verilator，这样源文件路径就是相对的
cd unittests/Memory && verilator -Wall --trace -cc main_tb.v output.v RAM32M.v RAM32X1D.v RAM64X1D.v RAM128X1D.v RAM256X1D.v RAM64M.v RAMB36E2.v --exe main.cpp --top-module main_tb -Wno-UNUSED -Wno-PINCONNECTEMPTY -Wno-DECLFILENAME -Wno-MULTITOP -Wno-PINMISSING --Mdir obj_dir && make -C obj_dir -f Vmain_tb.mk && ./obj_dir/Vmain_tb

cd ../..