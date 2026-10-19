# Vitis HLS C synthesis of the pre-decoder kernel. Author: Nasir Ali, C-DAC Noida.
# usage (from hls/synth): vitis_hls -f run_hls.tcl
open_project -reset predecoder_hls
set_top predecoder
add_files ../predecoder.cpp -cflags "-std=c++17 -I.."
open_solution -reset solution1 -flow_target vitis
set_part xcu55c-fsvh2892-2L-e
create_clock -period 3.33 -name default
csynth_design
exit
