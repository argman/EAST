CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LDSTACK = -Wl,-stack_size,1000000
	TMPFLAGS = $(shell python3-config --ldflags)
	LDFLAGS := $(filter-out $(LDSTACK), $(TMPFLAGS))
	DEPS = lanms.h $(shell find include -type f)
else
	LDFLAGS = $(shell python3-config --ldflags)
	DEPS = lanms.h $(shell find include -xtype f)
endif

CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
