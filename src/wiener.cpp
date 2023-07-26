#include "wiener.hpp"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

// Wiener filter function
std::array<Eigen::Tensor3dXcf, 4> umxcpp::wiener_filter(
    const Eigen::Tensor3dXcf &mix_stft,
    const std::array<Eigen::Tensor3dXcf, 4> &targets_spectrograms,
    int iterations,
    bool softmask,
    bool residual,
    float scale_factor,
    float eps
) {
    // copy target_spectrograms to return the same thing
    // before implementing this function
    std::array<Eigen::Tensor3dXcf, 4> result = targets_spectrograms;
    return result;
}
