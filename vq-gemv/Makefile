CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-variable
FRAMEWORKS := -framework Metal -framework Foundation

vqgemv: src/main.mm
	$(CXX) $(CXXFLAGS) $< $(FRAMEWORKS) -o $@

verify_real: src/verify_real.mm
	$(CXX) $(CXXFLAGS) $< $(FRAMEWORKS) -o $@

clean:
	rm -f vqgemv verify_real

.PHONY: clean
