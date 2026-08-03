CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./inc -O3

SRC_DIR = src
INC_DIR = inc
EXAMPLES_DIR = examples
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:.cpp=.o)

EXAMPLE_SRCS = $(wildcard $(EXAMPLES_DIR)/*/main.cpp)
TARGETS = $(patsubst $(EXAMPLES_DIR)/%/main.cpp,$(BIN_DIR)/%.exe,$(EXAMPLE_SRCS))

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%.exe: $(EXAMPLES_DIR)/%/main.cpp $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS)
	rm -rf $(BIN_DIR)
