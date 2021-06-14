#include <CL/sycl.hpp>
#include <vector>
#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
		
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
using namespace sycl;
using namespace oneapi::mkl::sparse;
namespace py = pybind11;

/* total 3 bindings need to be created

set b-vec

matrix handles

solve jacobi


*/

struct csr_matrix_elements
{
	std::vector<int> row;
	std::vector<int>col;
	std::vector<double> values;
	matrix_handle_t matrix_handle;
	int size;

};

struct matrix_elements_for_jacobi {
	csr_matrix_elements D_inv;
	csr_matrix_elements R_omega;
};

matrix_elements_for_jacobi jacobi_matrix;
std::vector<double>b_vec;


//void create_sparse_matrix_handle(std::vector<int>& row, std::vector<int>& col, std::vector<double>& vals , int&size,
//	std::string &mat_name) {
	// Accept the numpy array and then convert it to C++ vectors.
void create_sparse_matrix_handle(std::vector<int>& row, std::vector<int>& col, std::vector<double>& vals, int& size,
	std::string& mat_name) {

	//copy the py vectors to std::vectors


	if (mat_name == "D_inv") {
		jacobi_matrix.D_inv.col = col;
		jacobi_matrix.D_inv.row = row;
		jacobi_matrix.D_inv.values = vals;
		jacobi_matrix.D_inv.size = size;
		init_matrix_handle(&jacobi_matrix.D_inv.matrix_handle);
		set_csr_data(jacobi_matrix.D_inv.matrix_handle, jacobi_matrix.D_inv.size, jacobi_matrix.D_inv.size,
			oneapi::mkl::index_base::zero, jacobi_matrix.D_inv.row.data(), jacobi_matrix.D_inv.col.data(),
			jacobi_matrix.D_inv.values.data());
	}
	else {		
		jacobi_matrix.R_omega.col = col;
		jacobi_matrix.R_omega.row = row;
		jacobi_matrix.R_omega.values = vals;
		jacobi_matrix.R_omega.size = size;
		init_matrix_handle(&jacobi_matrix.R_omega.matrix_handle);
		set_csr_data(jacobi_matrix.R_omega.matrix_handle, jacobi_matrix.R_omega.size, jacobi_matrix.R_omega.size,
			oneapi::mkl::index_base::zero, jacobi_matrix.R_omega.row.data(), jacobi_matrix.R_omega.col.data(),
			jacobi_matrix.R_omega.values.data());
	}
}

void set_b_vector(std::vector<double>& b_vector) {
	b_vec = b_vector;
}

std::vector<double> jacobirelaxation() {
	cl::sycl::queue q;
	const float omega = 4.0 / 5.0;
	std::vector<double> ans(jacobi_matrix.D_inv.size, 0.0);
	std::vector<double> prod_1(jacobi_matrix.D_inv.size, 0.0);
	std::vector<double> prod_2(jacobi_matrix.D_inv.size, 0.0);

	cl::sycl::event R_V_mul_done = cl::sycl::event();
	cl::sycl::event D_inv_F_done = cl::sycl::event(); // Replace it by a vector to vector multiplication
	cl::sycl::event add_done = cl::sycl::event();

	for (int i = 0; i < 10; i++) {
		R_V_mul_done = gemv(q, oneapi::mkl::transpose::nontrans, 1.0, jacobi_matrix.R_omega.matrix_handle, 
			ans.data(), 0.0, prod_1.data());
		D_inv_F_done = gemv(q, oneapi::mkl::transpose::nontrans, omega, jacobi_matrix.D_inv.matrix_handle,
			b_vec.data(), 0.0, prod_2.data());
		add_done = oneapi::mkl::vm::add(q, jacobi_matrix.D_inv.size, prod_1.data(), prod_2.data(), ans.data(),
			{ R_V_mul_done , D_inv_F_done });
	}
	return ans;
}

PYBIND11_MODULE(smoother_sycl, m) {
	py::bind_vector<std::vector>
}