

TF_CFLAGS := `python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'` 
TF_LFLAGS := `python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`

USEGPU := -D GOOGLE_CUDA=1

ALLIBS := $(patsubst %_module.cc, %.so, $(wildcard *_module.cc))

all: $(ALLIBS)
	
	
	
%.so: %_kernel.o %_module.o %_kernel.cu.o
	g++ -std=c++14 -shared -o $@ $^ $(TF_CFLAGS) $(USEGPU) -fPIC -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart $(TF_LFLAGS)
	ln -sf compiled/$@  ../$@ 

%_module.o: %_module.cc	
	g++ -std=c++14 -c -o $@ $< $(TF_CFLAGS) $(USEGPU) -fPIC -I/usr/local/cuda/include
	
%_kernel.o: %_kernel.cc
	g++ -std=c++14 -c -o $@ $< $(TF_CFLAGS) $(USEGPU) -fPIC -I/usr/local/cuda/include
	
%_kernel.cu.o: %_kernel.cu.cc
	nvcc -std=c++14 -c -o $@ $< $(TF_CFLAGS) $(USEGPU) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -L/usr/local/cuda/lib64 -lcudart $(TF_LFLAGS) -DNDEBUG
	
clean:
	rm -f $(ALLIBS)

.PRECIOUS: %.o
