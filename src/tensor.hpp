#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <vector>

namespace Eigen {
    // define Tensor3dXf, Tensor3dXcf for spectrograms etc.
    typedef Tensor<float, 3> Tensor3dXf;
    typedef Tensor<std::complex<float>, 3> Tensor3dXcf;

    typedef Tensor<float, 4> Tensor4dXf;
    typedef Tensor<std::complex<float>, 4> Tensor4dXcf;
}

#endif // TENSOR_HPP
