# Syntropy Core - FPGA Build System
# Supports Xilinx Ultrascale+ and Lattice iCE40 targets

# Default target
TARGET ?= xilinx
PROJECT_NAME = syntropy_core
TOP_MODULE = syntropy_core

# Tool paths
VIVADO_PATH ?= /opt/Xilinx/Vivado/2023.1/bin
ICECUBE_PATH ?= /opt/lattice/icecube2
YOSYS_PATH ?= /usr/bin

# Build directories
BUILD_DIR = build
SYNTH_DIR = $(BUILD_DIR)/synthesis
IMPL_DIR = $(BUILD_DIR)/implementation
BITSTREAM_DIR = $(BUILD_DIR)/bitstream

# Verilog sources
VERILOG_SOURCES = verilog/syntropy_core.v
VERILOG_TOP = verilog/syntropy_core.v

# Constraints
XILINX_CONSTRAINTS = constraints/xilinx.xdc
LATTICE_CONSTRAINTS = constraints/lattice.pcf

# Simulation
SIM_DIR = simulation
TESTBENCH = verilog/tb_syntropy_core.v

# Python simulation
PYTHON_SIM = simulation/syntropy_sim.py

# Default target
all: $(TARGET)

# Xilinx Ultrascale+ target
xilinx: setup
	@echo "🔧 Building for Xilinx Ultrascale+"
	@mkdir -p $(SYNTH_DIR) $(IMPL_DIR) $(BITSTREAM_DIR)
	@$(VIVADO_PATH)/vivado -mode batch -source scripts/xilinx_synth.tcl \
		-tclargs $(PROJECT_NAME) $(VERILOG_TOP) $(XILINX_CONSTRAINTS) $(SYNTH_DIR)
	@$(VIVADO_PATH)/vivado -mode batch -source scripts/xilinx_impl.tcl \
		-tclargs $(PROJECT_NAME) $(SYNTH_DIR) $(IMPL_DIR) $(BITSTREAM_DIR)
	@echo "✅ Xilinx build complete"

# Lattice iCE40 target
lattice: setup
	@echo "🔧 Building for Lattice iCE40"
	@mkdir -p $(SYNTH_DIR) $(BITSTREAM_DIR)
	@$(YOSYS_PATH)/yosys -c scripts/lattice_synth.tcl \
		-p "synth_ice40 -top $(TOP_MODULE) -json $(SYNTH_DIR)/$(PROJECT_NAME).json" \
		$(VERILOG_SOURCES)
	@$(ICECUBE_PATH)/bin/linux64/nextpnr-ice40 --hx8k --package tq144:4k \
		--json $(SYNTH_DIR)/$(PROJECT_NAME).json --pcf $(LATTICE_CONSTRAINTS) \
		--asc $(BITSTREAM_DIR)/$(PROJECT_NAME).asc
	@$(ICECUBE_PATH)/bin/linux64/icepack $(BITSTREAM_DIR)/$(PROJECT_NAME).asc \
		$(BITSTREAM_DIR)/$(PROJECT_NAME).bin
	@echo "✅ Lattice build complete"

# Simulation targets
sim: sim-verilog sim-python

sim-verilog: setup
	@echo "🧪 Running Verilog simulation"
	@mkdir -p $(SIM_DIR)
	@iverilog -o $(SIM_DIR)/sim $(VERILOG_SOURCES) $(TESTBENCH)
	@vvp $(SIM_DIR)/sim
	@echo "✅ Verilog simulation complete"

sim-python: setup
	@echo "🧪 Running Python simulation"
	@cd $(SIM_DIR) && python3 $(PYTHON_SIM) --nodes 4 --duration 60
	@echo "✅ Python simulation complete"

# Hardware testing
test: test-fpga

test-fpga:
	@echo "🔬 Testing FPGA hardware"
	@python3 hardware/test_fpga.py --device /dev/ttyUSB0 --duration 30
	@echo "✅ FPGA testing complete"

# Performance benchmarks
benchmark: benchmark-sim benchmark-hw

benchmark-sim:
	@echo "📊 Running simulation benchmarks"
	@cd $(SIM_DIR) && python3 $(PYTHON_SIM) --nodes 8 --duration 300 --no-save
	@echo "✅ Simulation benchmarks complete"

benchmark-hw:
	@echo "📊 Running hardware benchmarks"
	@python3 hardware/test_fpga.py --device /dev/ttyUSB0 --duration 120 --output benchmark_results.json
	@echo "✅ Hardware benchmarks complete"

# Code generation
gen-verilog:
	@echo "⚙️  Generating Verilog from Python"
	@python3 scripts/generate_verilog.py --output verilog/generated/
	@echo "✅ Verilog generation complete"

gen-testbench:
	@echo "⚙️  Generating testbench"
	@python3 scripts/generate_testbench.py --output verilog/tb_generated.v
	@echo "✅ Testbench generation complete"

# Documentation
docs: docs-html docs-pdf

docs-html:
	@echo "📚 Generating HTML documentation"
	@mkdir -p docs/html
	@python3 scripts/generate_docs.py --format html --output docs/html/
	@echo "✅ HTML documentation complete"

docs-pdf:
	@echo "📚 Generating PDF documentation"
	@mkdir -p docs/pdf
	@python3 scripts/generate_docs.py --format pdf --output docs/pdf/
	@echo "✅ PDF documentation complete"

# Setup build environment
setup:
	@echo "🔧 Setting up build environment"
	@mkdir -p $(BUILD_DIR) $(SYNTH_DIR) $(IMPL_DIR) $(BITSTREAM_DIR) $(SIM_DIR)
	@mkdir -p constraints scripts docs/html docs/pdf
	@echo "✅ Build environment ready"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts"
	@rm -rf $(BUILD_DIR) $(SIM_DIR)/sim $(SIM_DIR)/*.vcd
	@rm -rf docs/html docs/pdf
	@rm -f *.log *.jou
	@echo "✅ Clean complete"

# Clean everything including generated files
distclean: clean
	@echo "🧹 Deep cleaning"
	@rm -rf verilog/generated/
	@rm -f verilog/tb_generated.v
	@rm -f *.json *.csv
	@echo "✅ Deep clean complete"

# Install dependencies
install-deps:
	@echo "📦 Installing dependencies"
	@pip3 install -r requirements.txt
	@echo "✅ Dependencies installed"

# Format code
format:
	@echo "🎨 Formatting code"
	@python3 -m black simulation/ hardware/ scripts/
	@echo "✅ Code formatted"

# Lint code
lint:
	@echo "🔍 Linting code"
	@python3 -m flake8 simulation/ hardware/ scripts/
	@echo "✅ Code linted"

# Run all checks
check: format lint sim test
	@echo "✅ All checks passed"

# Help
help:
	@echo "Syntropy Core Build System"
	@echo "========================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build for default target (xilinx)"
	@echo "  xilinx       - Build for Xilinx Ultrascale+"
	@echo "  lattice      - Build for Lattice iCE40"
	@echo "  sim          - Run all simulations"
	@echo "  sim-verilog  - Run Verilog simulation"
	@echo "  sim-python   - Run Python simulation"
	@echo "  test         - Run hardware tests"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  gen-verilog  - Generate Verilog from Python"
	@echo "  gen-testbench- Generate testbench"
	@echo "  docs         - Generate documentation"
	@echo "  setup        - Setup build environment"
	@echo "  clean        - Clean build artifacts"
	@echo "  distclean    - Deep clean"
	@echo "  install-deps - Install Python dependencies"
	@echo "  format       - Format code"
	@echo "  lint         - Lint code"
	@echo "  check        - Run all checks"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  TARGET       - Target FPGA (xilinx|lattice)"
	@echo "  VIVADO_PATH  - Path to Vivado installation"
	@echo "  ICECUBE_PATH - Path to iCEcube2 installation"
	@echo ""
	@echo "Examples:"
	@echo "  make xilinx VIVADO_PATH=/opt/Xilinx/Vivado/2023.1/bin"
	@echo "  make lattice"
	@echo "  make sim-python --nodes 8 --duration 120"
	@echo "  make test-fpga --device /dev/ttyUSB0"

# Phony targets
.PHONY: all xilinx lattice sim sim-verilog sim-python test test-fpga
.PHONY: benchmark benchmark-sim benchmark-hw gen-verilog gen-testbench
.PHONY: docs docs-html docs-pdf setup clean distclean install-deps
.PHONY: format lint check help