# Xilinx Synthesis Script for Syntropy Core
# Generates optimized netlist for Xilinx Ultrascale+ FPGAs

# Script arguments
set project_name [lindex $argv 0]
set verilog_top [lindex $argv 1]
set constraints_file [lindex $argv 2]
set output_dir [lindex $argv 3]

# Create project
create_project $project_name $output_dir -part xczu9eg-ffvb1156-2-e -force

# Add Verilog sources
add_files -norecurse $verilog_top
set_property top syntropy_core [current_fileset]

# Add constraints
if {[file exists $constraints_file]} {
    add_files -fileset constrs_1 -norecurse $constraints_file
}

# Set synthesis options for performance
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-retiming} -objects [get_runs synth_1]

# Set synthesis strategy for performance
set_property strategy Performance_Explore [get_runs synth_1]

# Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis results
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "Synthesis failed"
}

# Generate reports
open_run synth_1
report_utilization -file $output_dir/utilization_synth.rpt
report_timing -file $output_dir/timing_synth.rpt
report_power -file $output_dir/power_synth.rpt

# Export netlist
write_verilog -mode funcsim $output_dir/${project_name}_synth.v
write_edif $output_dir/${project_name}_synth.edif

puts "✅ Synthesis complete: $output_dir"