#include <iostream>

// The SYCL header
#include <CL/sycl.hpp>

// <<Declare vector_addition kernel function object>>
class vector_addition;

int main(int, char**) {
	// <<Setup host memory>>
	// define input vectors
	cl::sycl::float4 a = { 1.0, 2.0, 3.0, 4.0 };
	cl::sycl::float4 b = { 4.0, 3.0, 2.0, 1.0 };
	// define output vector
	cl::sycl::float4 c = { 0.0, 0.0, 0.0, 0.0 };
	
	// <<Setup SYCL queue>>	
	cl::sycl::default_selector device_selector;	
	cl::sycl::queue queue(device_selector);
	
	std::cout << "Running on "
	          << queue.get_device().get_info<cl::sycl::info::device::name>()
			  << "\n";
	
	// Begin SYCL scope
	{
		// <<Setup device memory>>
		// define input buffers
		cl::sycl::buffer<cl::sycl::float4, 1> a_sycl(&a, cl::sycl::range<1>(1));
		cl::sycl::buffer<cl::sycl::float4, 1> b_sycl(&b, cl::sycl::range<1>(1));
		// define output buffer
		cl::sycl::buffer<cl::sycl::float4, 1> c_sycl(&c, cl::sycl::range<1>(1));
		
		// Submit a command group functor for execution on a queue. This functor
        // encapsulates the kernel and the data needed for its execution.
	    queue.submit([&] (cl::sycl::handler& cgh) {
			// <<Request device memory access>>
			// read accessors
			auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
			auto b_acc = b_syc1.get_access<cl::sycl::access::mode::read>(cgh);
			// discard_write accessor
			auto c_acc = c_syc1.get_access<cl::sycl::access::mode::discard_write>(cgh);
			
			// Enqueue the kernel for execution using the `single_task` API
			cgh.single_task<class vector_addition>([=] () {
				// <<Complete the vector addition computation>>
                // calculate: c = a+b;
				c_acc[0] = a_acc[0] + b_acc[0];
			});
		});
	}
	// End SYCL scope
	
	std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
	          << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
			  << "------------------\n"
			  << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }\n"
			  << std::endl;

    return 0;
}