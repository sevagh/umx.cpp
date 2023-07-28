#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace Eigen
{
// define Tensor3dXf, Tensor3dXcf for spectrograms etc.
typedef Tensor<float, 3> Tensor3dXf;
typedef Tensor<std::complex<float>, 3> Tensor3dXcf;
} // namespace Eigen

#endif // TENSOR_HPP
