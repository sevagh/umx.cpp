#ifndef WIENER_HPP
#define WIENER_HPP

#include <array>
#include <string>
#include "dsp.hpp"

namespace umxcpp {
     std::array<StereoSpectrogramC, 4> wiener_filter(
        const StereoSpectrogramC &spectrogram,
        const StereoSpectrogramC (&targets)[4]
     );

}

#endif // WIENER_HPP
