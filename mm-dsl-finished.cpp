#include "builder/dyn_var.h"
#include "builder/static_var.h"
#include "builder/builder.h"
#include "blocks/c_code_generator.h"
#include "blocks/for_loop_finder.h"
#include "blocks/rce.h"
#include "pipeline/graphit.h"
using builder::dyn_var;
using builder::static_var;

/* RUNTIME FUNCTION DECLARTION */
namespace runtime {
	dyn_var<void*(int)> malloc("runtime::malloc");
	dyn_var<void*(int)> device_malloc("runtime::device_malloc");
	dyn_var<void(void*, void*, int)> to_device("runtime::to_device");
	dyn_var<void(void*, void*, int)> to_host("runtime::to_host");
	dyn_var<void(void*, void*, int)> memcpy("runtime::memcpy");
}
#define CUDA_KERNEL "run_on_device"

/* DUAL ARRAY TYPE DECLARATION */
#define HOST (1)
#define DEVICE (0)
// Variable to track where the code is running currently
int current_context;

template <typename T> 
struct dual_array_index;

// Data type for an array that can move between Host and GPU
template <typename T>
struct dual_array {
	dyn_var<int> size;
	dyn_var<T*> host_buffer;
	dyn_var<T*> device_buffer;
	// Static variable to determine where the array currently resides
	static_var<int> current = 0;

	// Functions to allocate and move the buffers around
	void allocate(dyn_var<int> s) {
		size = s;
		host_buffer = runtime::malloc((int)sizeof(T) * size);
		device_buffer = runtime::device_malloc((int)sizeof(T) * size);
		current = HOST;
	}
	void move_to_device(void) {
		if (current == DEVICE) return;
		runtime::to_device(device_buffer, host_buffer, (int)sizeof(T) * size);
		current = DEVICE;
	}
	void move_to_host(void) {
		if (current == HOST) return;
		runtime::to_host(host_buffer, device_buffer, (int)sizeof(T) * size);
		current = HOST;
	}
	struct dual_array_index<T> operator[] (dyn_var<int> i) {
		struct dual_array_index<T> di(this, i);
		return di;
	}
};
// Supporting data type for dual_array 
template <typename T> 
struct dual_array_index {
	dual_array<T> *array;
	dyn_var<int> index;
	dual_array_index(dual_array<T>* a, dyn_var<int> i): array(a), index(i) {}
	
	// Function to return the value specialized based on context and where the 
	// array is located
	dyn_var<T> get(void) {
		if (array->current == HOST && current_context == HOST) {
			return array->host_buffer[index];
		}
		if (array->current == DEVICE && current_context == DEVICE) {
			return array->device_buffer[index];
		}	
		if (array->current == DEVICE && current_context == HOST) {
			dyn_var<T> temp;
			runtime::to_host(&temp, &(array->device_buffer[index]), (int)sizeof(T));
			return temp;	
		}	
		if (array->current == HOST && current_context == DEVICE) {
			assert(false && "Cannot move host array to device from device. Please insert moves at appropriate places");	
			dyn_var<T> ret = 0;
			return ret;
		}
	}
	// Function to update the value specialized based on context and where the 
	// array is located
	void operator= (const dyn_var<T> rhs) {
		if (array->current == HOST && current_context == HOST) {
			array->host_buffer[index] = rhs;
			return;
		}
		if (array->current == DEVICE && current_context == DEVICE) {
			array->device_buffer[index] = rhs;
			return;
		}	
		if (array->current == DEVICE && current_context == HOST) {
			dyn_var<T> temp = rhs;
			runtime::to_device(&(array->device_buffer[index]), &temp, (int)sizeof(T));
			return;	
		}	
		if (array->current == HOST && current_context == DEVICE) {
			assert(false && "Cannot move host array to device from device. Please insert moves at appropriate places");	
			return;
		}
	}
};

/* MATRIX VECTOR PRODUCT IMPLEMENTATION */
void mmvp(dyn_var<int> n, dual_array<float> &M, dual_array<float> &x, dual_array<float> &y, const int mode) {
	if (mode == HOST) {	
		// Host side implementation of y = M * x	
		M.move_to_host();
		x.move_to_host();
		y.move_to_host();
		for (dyn_var<int> r = 0; r < n; r = r + 1) {
			y[r] = 0.0f;
			for (dyn_var<int> c = 0; c < n; c = c + 1) {
				y[r] = y[r].get() + M[r * n + c].get() * x[c].get();
			}	
		}
	} else {
		// Device side implementation of y = M * x	
		M.move_to_device();
		x.move_to_device();
		y.move_to_device();
		current_context = DEVICE;
		dyn_var<int> num_cta = (n + 511) / 512;

		builder::annotate(CUDA_KERNEL);
		for (dyn_var<int> cta = 0; cta < num_cta; cta = cta + 1) {
			for(dyn_var<int> tid = 0; tid < 512; tid = tid + 1) {
				dyn_var<int> r = cta * 512 + tid;
				if (r < n) {
					y[r] = 0.0f;
					for (dyn_var<int> c = 0; c < n; c = c + 1) {
						y[r] = y[r].get() + M[r * n + c].get() * x[c].get();
					}	
				}
			}
		}
		current_context = HOST;
	}
}


/* TEST APP WRITTEN WITH DSL */
dyn_var<int*> test(dyn_var<int> n, dyn_var<float*> _M, dyn_var<float*> _x) {
	
	current_context = HOST;
	// Allocate dual buffers for all arrays
	dual_array<float> x, y;
	x.allocate(n);
	y.allocate(n);
	runtime::memcpy(x.host_buffer, _x, (int)sizeof(float) * n);
	
	dual_array<float> M;
	M.allocate(n * n);
	runtime::memcpy(M.host_buffer, _M, (int)sizeof(float) * n * n);	

	// 4 calls to multiplication back and forth
	for (static_var<int> iter = 0; iter < 4; iter = iter + 1) {
		if (iter % 2 == 0)
			mmvp(n, M, x, y, HOST);
		else
			mmvp(n, M, y, x, DEVICE);
		
	}
	x.move_to_host();
	return x.host_buffer;
}

/* MAIN DRIVER FUNCTION TO GENERATE CODE */
int main(int argc, char* argv[]) {
	builder::builder_context context;
	context.run_rce = true;
	auto ast = context.extract_function_ast(test, "mmvp_app");

	block::block::Ptr kernel;
	std::vector<block::decl_stmt::Ptr> new_decls;
	while (kernel = pipeline::extract_single_cuda(block::to<block::func_decl>(ast)->body, new_decls)) {
		block::c_code_generator::generate_code(kernel, std::cout);
	}

	block::c_code_generator::generate_code(ast, std::cout);
	return 0;
}
