# Xilinx Constraints for Syntropy Core
# Optimized for Xilinx Ultrascale+ FPGAs

# Clock constraints
create_clock -period 10.000 -name clk [get_ports clk]
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

# Reset constraints
set_property PACKAGE_PIN AA4 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# Sensor input constraints (ADC interface)
set_property PACKAGE_PIN AB5 [get_ports {voltage_drift[0]}]
set_property PACKAGE_PIN AB6 [get_ports {voltage_drift[1]}]
set_property PACKAGE_PIN AB7 [get_ports {voltage_drift[2]}]
set_property PACKAGE_PIN AB8 [get_ports {voltage_drift[3]}]
set_property PACKAGE_PIN AB9 [get_ports {voltage_drift[4]}]
set_property PACKAGE_PIN AB10 [get_ports {voltage_drift[5]}]
set_property PACKAGE_PIN AB11 [get_ports {voltage_drift[6]}]
set_property PACKAGE_PIN AB12 [get_ports {voltage_drift[7]}]
set_property PACKAGE_PIN AB13 [get_ports {voltage_drift[8]}]
set_property PACKAGE_PIN AB14 [get_ports {voltage_drift[9]}]
set_property PACKAGE_PIN AB15 [get_ports {voltage_drift[10]}]
set_property PACKAGE_PIN AB16 [get_ports {voltage_drift[11]}]
set_property PACKAGE_PIN AB17 [get_ports {voltage_drift[12]}]
set_property PACKAGE_PIN AB18 [get_ports {voltage_drift[13]}]
set_property PACKAGE_PIN AB19 [get_ports {voltage_drift[14]}]
set_property PACKAGE_PIN AB20 [get_ports {voltage_drift[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {voltage_drift[*]}]

# Packet loss inputs
set_property PACKAGE_PIN AC5 [get_ports {packet_loss[0]}]
set_property PACKAGE_PIN AC6 [get_ports {packet_loss[1]}]
set_property PACKAGE_PIN AC7 [get_ports {packet_loss[2]}]
set_property PACKAGE_PIN AC8 [get_ports {packet_loss[3]}]
set_property PACKAGE_PIN AC9 [get_ports {packet_loss[4]}]
set_property PACKAGE_PIN AC10 [get_ports {packet_loss[5]}]
set_property PACKAGE_PIN AC11 [get_ports {packet_loss[6]}]
set_property PACKAGE_PIN AC12 [get_ports {packet_loss[7]}]
set_property PACKAGE_PIN AC13 [get_ports {packet_loss[8]}]
set_property PACKAGE_PIN AC14 [get_ports {packet_loss[9]}]
set_property PACKAGE_PIN AC15 [get_ports {packet_loss[10]}]
set_property PACKAGE_PIN AC16 [get_ports {packet_loss[11]}]
set_property PACKAGE_PIN AC17 [get_ports {packet_loss[12]}]
set_property PACKAGE_PIN AC18 [get_ports {packet_loss[13]}]
set_property PACKAGE_PIN AC19 [get_ports {packet_loss[14]}]
set_property PACKAGE_PIN AC20 [get_ports {packet_loss[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {packet_loss[*]}]

# Temperature variance inputs
set_property PACKAGE_PIN AD5 [get_ports {temp_variance[0]}]
set_property PACKAGE_PIN AD6 [get_ports {temp_variance[1]}]
set_property PACKAGE_PIN AD7 [get_ports {temp_variance[2]}]
set_property PACKAGE_PIN AD8 [get_ports {temp_variance[3]}]
set_property PACKAGE_PIN AD9 [get_ports {temp_variance[4]}]
set_property PACKAGE_PIN AD10 [get_ports {temp_variance[5]}]
set_property PACKAGE_PIN AD11 [get_ports {temp_variance[6]}]
set_property PACKAGE_PIN AD12 [get_ports {temp_variance[7]}]
set_property PACKAGE_PIN AD13 [get_ports {temp_variance[8]}]
set_property PACKAGE_PIN AD14 [get_ports {temp_variance[9]}]
set_property PACKAGE_PIN AD15 [get_ports {temp_variance[10]}]
set_property PACKAGE_PIN AD16 [get_ports {temp_variance[11]}]
set_property PACKAGE_PIN AD17 [get_ports {temp_variance[12]}]
set_property PACKAGE_PIN AD18 [get_ports {temp_variance[13]}]
set_property PACKAGE_PIN AD19 [get_ports {temp_variance[14]}]
set_property PACKAGE_PIN AD20 [get_ports {temp_variance[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {temp_variance[*]}]

# Phase jitter inputs
set_property PACKAGE_PIN AE5 [get_ports {phase_jitter[0]}]
set_property PACKAGE_PIN AE6 [get_ports {phase_jitter[1]}]
set_property PACKAGE_PIN AE7 [get_ports {phase_jitter[2]}]
set_property PACKAGE_PIN AE8 [get_ports {phase_jitter[3]}]
set_property PACKAGE_PIN AE9 [get_ports {phase_jitter[4]}]
set_property PACKAGE_PIN AE10 [get_ports {phase_jitter[5]}]
set_property PACKAGE_PIN AE11 [get_ports {phase_jitter[6]}]
set_property PACKAGE_PIN AE12 [get_ports {phase_jitter[7]}]
set_property PACKAGE_PIN AE13 [get_ports {phase_jitter[8]}]
set_property PACKAGE_PIN AE14 [get_ports {phase_jitter[9]}]
set_property PACKAGE_PIN AE15 [get_ports {phase_jitter[10]}]
set_property PACKAGE_PIN AE16 [get_ports {phase_jitter[11]}]
set_property PACKAGE_PIN AE17 [get_ports {phase_jitter[12]}]
set_property PACKAGE_PIN AE18 [get_ports {phase_jitter[13]}]
set_property PACKAGE_PIN AE19 [get_ports {phase_jitter[14]}]
set_property PACKAGE_PIN AE20 [get_ports {phase_jitter[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {phase_jitter[*]}]

# Control inputs
set_property PACKAGE_PIN AF5 [get_ports {threshold[0]}]
set_property PACKAGE_PIN AF6 [get_ports {threshold[1]}]
set_property PACKAGE_PIN AF7 [get_ports {threshold[2]}]
set_property PACKAGE_PIN AF8 [get_ports {threshold[3]}]
set_property PACKAGE_PIN AF9 [get_ports {threshold[4]}]
set_property PACKAGE_PIN AF10 [get_ports {threshold[5]}]
set_property PACKAGE_PIN AF11 [get_ports {threshold[6]}]
set_property PACKAGE_PIN AF12 [get_ports {threshold[7]}]
set_property PACKAGE_PIN AF13 [get_ports {threshold[8]}]
set_property PACKAGE_PIN AF14 [get_ports {threshold[9]}]
set_property PACKAGE_PIN AF15 [get_ports {threshold[10]}]
set_property PACKAGE_PIN AF16 [get_ports {threshold[11]}]
set_property PACKAGE_PIN AF17 [get_ports {threshold[12]}]
set_property PACKAGE_PIN AF18 [get_ports {threshold[13]}]
set_property PACKAGE_PIN AF19 [get_ports {threshold[14]}]
set_property PACKAGE_PIN AF20 [get_ports {threshold[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {threshold[*]}]

set_property PACKAGE_PIN AG5 [get_ports enable_learning]
set_property IOSTANDARD LVCMOS33 [get_ports enable_learning]

set_property PACKAGE_PIN AG6 [get_ports mesh_broadcast_en]
set_property IOSTANDARD LVCMOS33 [get_ports mesh_broadcast_en]

# Output constraints
set_property PACKAGE_PIN AH5 [get_ports stable]
set_property IOSTANDARD LVCMOS33 [get_ports stable]

set_property PACKAGE_PIN AH6 [get_ports {flux_value[0]}]
set_property PACKAGE_PIN AH7 [get_ports {flux_value[1]}]
set_property PACKAGE_PIN AH8 [get_ports {flux_value[2]}]
set_property PACKAGE_PIN AH9 [get_ports {flux_value[3]}]
set_property PACKAGE_PIN AH10 [get_ports {flux_value[4]}]
set_property PACKAGE_PIN AH11 [get_ports {flux_value[5]}]
set_property PACKAGE_PIN AH12 [get_ports {flux_value[6]}]
set_property PACKAGE_PIN AH13 [get_ports {flux_value[7]}]
set_property PACKAGE_PIN AH14 [get_ports {flux_value[8]}]
set_property PACKAGE_PIN AH15 [get_ports {flux_value[9]}]
set_property PACKAGE_PIN AH16 [get_ports {flux_value[10]}]
set_property PACKAGE_PIN AH17 [get_ports {flux_value[11]}]
set_property PACKAGE_PIN AH18 [get_ports {flux_value[12]}]
set_property PACKAGE_PIN AH19 [get_ports {flux_value[13]}]
set_property PACKAGE_PIN AH20 [get_ports {flux_value[14]}]
set_property PACKAGE_PIN AJ5 [get_ports {flux_value[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {flux_value[*]}]

set_property PACKAGE_PIN AJ6 [get_ports {correction_vector[0]}]
set_property PACKAGE_PIN AJ7 [get_ports {correction_vector[1]}]
set_property PACKAGE_PIN AJ8 [get_ports {correction_vector[2]}]
set_property PACKAGE_PIN AJ9 [get_ports {correction_vector[3]}]
set_property PACKAGE_PIN AJ10 [get_ports {correction_vector[4]}]
set_property PACKAGE_PIN AJ11 [get_ports {correction_vector[5]}]
set_property PACKAGE_PIN AJ12 [get_ports {correction_vector[6]}]
set_property PACKAGE_PIN AJ13 [get_ports {correction_vector[7]}]
set_property PACKAGE_PIN AJ14 [get_ports {correction_vector[8]}]
set_property PACKAGE_PIN AJ15 [get_ports {correction_vector[9]}]
set_property PACKAGE_PIN AJ16 [get_ports {correction_vector[10]}]
set_property PACKAGE_PIN AJ17 [get_ports {correction_vector[11]}]
set_property PACKAGE_PIN AJ18 [get_ports {correction_vector[12]}]
set_property PACKAGE_PIN AJ19 [get_ports {correction_vector[13]}]
set_property PACKAGE_PIN AJ20 [get_ports {correction_vector[14]}]
set_property PACKAGE_PIN AK5 [get_ports {correction_vector[15]}]

set_property IOSTANDARD LVCMOS33 [get_ports {correction_vector[*]}]

# Status outputs
set_property PACKAGE_PIN AK6 [get_ports learning_active]
set_property IOSTANDARD LVCMOS33 [get_ports learning_active]

set_property PACKAGE_PIN AK7 [get_ports mesh_broadcast_req]
set_property IOSTANDARD LVCMOS33 [get_ports mesh_broadcast_req]

set_property PACKAGE_PIN AK8 [get_ports {iteration_count[0]}]
set_property PACKAGE_PIN AK9 [get_ports {iteration_count[1]}]
set_property PACKAGE_PIN AK10 [get_ports {iteration_count[2]}]
set_property PACKAGE_PIN AK11 [get_ports {iteration_count[3]}]
set_property PACKAGE_PIN AK12 [get_ports {iteration_count[4]}]
set_property PACKAGE_PIN AK13 [get_ports {iteration_count[5]}]
set_property PACKAGE_PIN AK14 [get_ports {iteration_count[6]}]
set_property PACKAGE_PIN AK15 [get_ports {iteration_count[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {iteration_count[*]}]

set_property PACKAGE_PIN AK16 [get_ports {state[0]}]
set_property PACKAGE_PIN AK17 [get_ports {state[1]}]
set_property PACKAGE_PIN AK18 [get_ports {state[2]}]
set_property PACKAGE_PIN AK19 [get_ports {state[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports {state[*]}]

set_property PACKAGE_PIN AK20 [get_ports error_flag]
set_property IOSTANDARD LVCMOS33 [get_ports error_flag]

# Timing constraints
set_input_delay -clock clk -min -add_delay 2.0 [get_ports {voltage_drift[*]}]
set_input_delay -clock clk -max -add_delay 4.0 [get_ports {voltage_drift[*]}]

set_input_delay -clock clk -min -add_delay 2.0 [get_ports {packet_loss[*]}]
set_input_delay -clock clk -max -add_delay 4.0 [get_ports {packet_loss[*]}]

set_input_delay -clock clk -min -add_delay 2.0 [get_ports {temp_variance[*]}]
set_input_delay -clock clk -max -add_delay 4.0 [get_ports {temp_variance[*]}]

set_input_delay -clock clk -min -add_delay 2.0 [get_ports {phase_jitter[*]}]
set_input_delay -clock clk -max -add_delay 4.0 [get_ports {phase_jitter[*]}]

set_output_delay -clock clk -min -add_delay 1.0 [get_ports {flux_value[*]}]
set_output_delay -clock clk -max -add_delay 3.0 [get_ports {flux_value[*]}]

set_output_delay -clock clk -min -add_delay 1.0 [get_ports {correction_vector[*]}]
set_output_delay -clock clk -max -add_delay 3.0 [get_ports {correction_vector[*]}]

# Power optimization
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# Clock domain crossing
set_clock_groups -asynchronous -group [get_clocks clk]

# False paths
set_false_path -from [get_ports rst_n] -to [get_ports {flux_value[*]}]
set_false_path -from [get_ports enable_learning] -to [get_ports {flux_value[*]}]

# Multicycle paths for learning
set_multicycle_path -setup 2 -from [get_ports enable_learning] -to [get_ports learning_active]
set_multicycle_path -hold 1 -from [get_ports enable_learning] -to [get_ports learning_active]

# High fanout nets
set_property MAX_FANOUT 100 [get_nets clk]
set_property MAX_FANOUT 50 [get_nets rst_n]

# Placement constraints for performance
set_property LOC SLICE_X0Y0 [get_cells syntropy_core_inst]
set_property LOC DSP48E2_X0Y0 [get_cells *mult*]

# Memory constraints
set_property RAM_STYLE BLOCK [get_cells *params*]
set_property ROM_STYLE BLOCK [get_cells *gradients*]

# Optimization constraints
set_property OPTIMIZATION_LEVEL 2 [get_runs synth_1]
set_property OPTIMIZATION_LEVEL 2 [get_runs impl_1]

# Retiming for performance
set_property STEPS.SYNTH_DESIGN.ARGS.MORE_OPTIONS {-retiming} [get_runs synth_1]

# Pipeline registers
set_property REGISTER_BALANCING Yes [get_runs synth_1]
set_property REGISTER_DUPLICATION Yes [get_runs synth_1]

# Power optimization
set_property POWER_OPTIMIZATION HIGH [get_runs impl_1]

# Debug constraints
set_property MARK_DEBUG true [get_nets flux_value]
set_property MARK_DEBUG true [get_nets stable]
set_property MARK_DEBUG true [get_nets correction_vector]