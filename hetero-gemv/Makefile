CXX      := clang++
CXXFLAGS := -std=c++17 -O3 -fobjc-arc -Wall -Wno-unused-variable -Wno-deprecated-declarations
FRAMEWORKS := -framework Metal -framework Foundation -framework Accelerate

hetero: src/main.mm src/cpu_gemv.h
	$(CXX) $(CXXFLAGS) src/main.mm $(FRAMEWORKS) -o $@

amx_probe: src/amx_probe.c
	cc -O2 $< -framework Accelerate -o $@

clean:
	rm -f hetero amx_probe

.PHONY: clean
