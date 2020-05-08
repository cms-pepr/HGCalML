

TF_CFLAGS := `python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'` 
TF_LFLAGS := `python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`


all: $(patsubst %_module.cc, %.so, $(wildcard *_module.cc))
	
	
%.so: %_kernel.cu.o %_kernel.o %_module.o
	g++ -std=c++11 -shared -o $@ $< $(TF_CFLAGS) -D GOOGLE_CUDA=1 -fPIC -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart $(TF_LFLAGS)
	ln -s compiled/$@  ../$@ 

%_module.o: %_module.cc	
	g++ -std=c++11 -c -o $@ $< $(TF_CFLAGS) -D GOOGLE_CUDA=1 -fPIC -I/usr/local/cuda/include
	
%_kernel.o: %_kernel.cc
	g++ -std=c++11 -c -o $@ $< $(TF_CFLAGS) -D GOOGLE_CUDA=1 -fPIC -I/usr/local/cuda/include
	
%_kernel.cu.o: %_kernel.cu.cc
	nvcc -std=c++11 -c -o $@ $< $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -L/usr/local/cuda/lib64 -lcudart $(TF_LFLAGS) -DNDEBUG
	

