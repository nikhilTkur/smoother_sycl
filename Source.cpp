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

struct csr_matrix_elements
{
	std::vector<int> row;
	std::vector<int>col;
	std::vector<double> values;
	matrix_handle_t matrix_handle;
	int size;
};


csr_matrix_elements create_sparse_matrix_handle(py::array_t<int>& row, py::array_t<int>& col, py::array_t<double>& vals, int& size) {

	std::vector<int> row_vec(row.data(), row.data() + row.size());
	std::vector<int> col_vec(col.data() , col.data() + col.size());
	std::vector<double> vals_vec(vals.data(), vals.data() +  vals.size());
	//std::memcpy(row_vec.data(), row.data(), row.size() * sizeof(int));
	//std::memcpy(col_vec.data(), col.data(), col.size() * sizeof(int));
	//std::memcpy(vals_vec.data(), vals.data(), vals.size() * sizeof(double));

	// Create a matrix entity 
	csr_matrix_elements matrix;
	matrix.row = row_vec;
	matrix.col = col_vec;
	matrix.values = vals_vec;
	matrix.size = size;
	init_matrix_handle(&matrix.matrix_handle);
	set_csr_data(matrix.matrix_handle, matrix.size, matrix.size, oneapi::mkl::index_base::zero, matrix.row.data(),
		matrix.col.data(), matrix.values.data());
	return matrix;
}

py::array_t<double> jacobirelaxation(csr_matrix_elements &D_inv , csr_matrix_elements &R_omega , py::array_t<double>& b) {

	std::vector<double> b_vec(b.data(), b.data() + b.size());
	cl::sycl::queue q;
	const float omega = 4.0 / 5.0;
	std::vector<double> ans(D_inv.size, 0.0);
	std::vector<double> prod_1(D_inv.size, 0.0);
	std::vector<double> prod_2(D_inv.size, 0.0);

	cl::sycl::event R_V_mul_done = cl::sycl::event();
	cl::sycl::event D_inv_F_done = cl::sycl::event(); // Replace it by a vector to vector multiplication
	cl::sycl::event add_done = cl::sycl::event();

	for (int i = 0; i < 10; i++) {
		R_V_mul_done = gemv(q, oneapi::mkl::transpose::nontrans, 1.0, R_omega.matrix_handle,
			ans.data(), 0.0, prod_1.data());
		D_inv_F_done = gemv(q, oneapi::mkl::transpose::nontrans, omega, D_inv.matrix_handle,
			b_vec.data(), 0.0, prod_2.data());
		add_done = oneapi::mkl::vm::add(q, D_inv.size, prod_1.data(), prod_2.data(), ans.data(),
			{ R_V_mul_done , D_inv_F_done });
	}
	// cast the result back to numpy array
	auto result = py::array_t<double>(ans.size());
	auto result_buffer = result.request();
	double* result_ptr = (double*)result_buffer.ptr;
	std::memcpy(result_ptr, ans.data(), ans.size() * sizeof(double));
	return result;
}

PYBIND11_MODULE(smoother_sycl, m) {
	py::class_<csr_matrix_elements>(m, "csr_martix_elements")
		.def(py::init<>());
	m.def("smoother_jacobi", &jacobirelaxation);
	m.def("create_csr_matrix", &create_sparse_matrix_handle);
	//py::bind_vector<std::vector>
}