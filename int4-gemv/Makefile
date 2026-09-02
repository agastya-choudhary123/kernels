CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-variable
FRAMEWORKS := -framework Metal -framework Foundation

int4gemv: src/main.mm
	$(CXX) $(CXXFLAGS) $< $(FRAMEWORKS) -o $@

run: int4gemv
	./int4gemv

clean:
	rm -f int4gemv

.PHONY: run clean
