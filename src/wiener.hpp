#ifndef WIENER_HPP
#define WIENER_HPP

#include <array>
#include <string>
#include "dsp.hpp"

namespace umxcpp {
     std::array<Eigen::Tensor3dXcf, 4> wiener_filter(
        const Eigen::Tensor3dXcf &spectrogram,
        const std::array<Eigen::Tensor3dXcf, 4> &targets_spectrograms,
        int iterations = 1,
        bool softmask = false,
        bool residual = false,
        float scale_factor = 10.0f,
        float eps = 1e-10f
     );
}

#endif // WIENER_HPP
