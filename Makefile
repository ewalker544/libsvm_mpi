CXX = mpic++
CFLAGS = -Wall -Wconversion -O3 -std=c++11
SHVER = 2
OS = $(shell uname)

all: svm-train-mpi

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-train-mpi: svm-train-mpi.cpp svm_par.o
	$(CXX) $(CFLAGS) svm-train-mpi.cpp svm_par.o -o svm-train-mpi -lm
svm_par.o: svm_par.cpp svm.h ThreadPool/SvmThreads.h ThreadPool/ThreadPool.h ThreadPool/CachePool.h ThreadPool/DistributedCache.h
	$(CXX) $(CFLAGS) -c svm_par.cpp
clean:
	rm -f *~ svm_par.o svm-train-mpi libsvm.so.$(SHVER)
