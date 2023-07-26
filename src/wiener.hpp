#ifndef WIENER_HPP
#define WIENER_HPP

#include "dsp.hpp"
#include <array>
#include <string>

namespace umxcpp {
const float WIENER_EPS = 1e-10f;
const float WIENER_SCALE_FACTOR = 10.0f;
const int WIENER_EM_BATCH_SIZE = 200;

std::array<Eigen::Tensor3dXcf, 4>
wiener_filter(const Eigen::Tensor3dXcf &spectrogram,
              const std::array<Eigen::Tensor3dXcf, 4> &targets_spectrograms);
} // namespace umxcpp

#endif // WIENER_HPP
