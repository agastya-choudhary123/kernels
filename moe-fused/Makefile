CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-variable
FRAMEWORKS := -framework Metal -framework Foundation

moefused: src/main.mm
	$(CXX) $(CXXFLAGS) $< $(FRAMEWORKS) -o $@

clean:
	rm -f moefused

.PHONY: clean
