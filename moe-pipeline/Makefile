CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-variable
FRAMEWORKS := -framework Metal -framework Foundation

moepipe: src/main.mm
	$(CXX) $(CXXFLAGS) $< $(FRAMEWORKS) -o $@

clean:
	rm -f moepipe

.PHONY: clean
