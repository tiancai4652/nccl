# Makefile for fake_cuda library
CXX = g++
CXXFLAGS = -fPIC -O2 -Wall -I../include

SRCFILES = fake_cuda.cc
OBJFILES = $(SRCFILES:.cc=.o)

LIBDIR = ../lib
LIBNAME_SHARED = libcudart.so
LIBNAME_STATIC = libcudart_static.a

all: $(LIBDIR)/$(LIBNAME_SHARED) $(LIBDIR)/$(LIBNAME_STATIC)

$(LIBDIR)/$(LIBNAME_SHARED): $(OBJFILES) | $(LIBDIR)
	$(CXX) -shared -o $@ $(OBJFILES)

$(LIBDIR)/$(LIBNAME_STATIC): $(OBJFILES) | $(LIBDIR)
	ar rcs $@ $(OBJFILES)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBDIR):
	mkdir -p $(LIBDIR)

clean:
	rm -f $(OBJFILES) $(LIBDIR)/$(LIBNAME_SHARED) $(LIBDIR)/$(LIBNAME_STATIC)

.PHONY: all clean
