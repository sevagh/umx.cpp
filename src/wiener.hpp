#ifndef WIENER_HPP
#define WIENER_HPP

#include <array>
#include <string>
#include "dsp.hpp"

namespace umxcpp {
   const float WIENER_EPS = 1e-10f;
   const float WIENER_SCALE_FACTOR = 10.0f;

   std::array<Eigen::Tensor3dXcf, 4> wiener_filter(
      const Eigen::Tensor3dXcf &spectrogram,
      const std::array<Eigen::Tensor3dXcf, 4> &targets_spectrograms
   );
}

#endif // WIENER_HPP
