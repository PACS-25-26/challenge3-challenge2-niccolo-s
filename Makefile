# Makefile for the parallel Jacobi solver (MPI + OpenMP)

# Compiler
CXX := mpicxx

# Directories
SRC_DIR := src
INC_DIR := include
EXT_DIR := external
OBJ_DIR := obj

# Target executable
TARGET := jacobi_solver

# Sources and objects
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Optimisation
OPT ?= -O2

# OpenMP: detect support and set flags accordingly
UNAME := $(shell uname)

# This check is needed make sure the code runs also on macOS machines
ifeq ($(UNAME), Darwin)
    # On macOS, clang does not support -fopenmp natively.
    # Check if libomp is installed via Homebrew (brew install libomp).
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(LIBOMP_PREFIX),)
        OMP_CXXFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
        OMP_LDFLAGS  := -L$(LIBOMP_PREFIX)/lib -lomp
    else
        $(warning OpenMP not found on macOS. Compiling without OpenMP support.)
        $(warning To enable OpenMP, run: brew install libomp)
        OMP_CXXFLAGS :=
        OMP_LDFLAGS  :=
    endif
else
    # Linux: g++ supports -fopenmp natively
    OMP_CXXFLAGS := -fopenmp
    OMP_LDFLAGS  := -fopenmp
endif

# Other flags
CXXFLAGS := -std=c++17 -Wall -Wextra -I$(INC_DIR) -I$(EXT_DIR) \
            $(OMP_CXXFLAGS) $(OPT)
LDFLAGS  := $(OMP_LDFLAGS)

# Rules
.PHONY: all clean

all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@

# Compile
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)