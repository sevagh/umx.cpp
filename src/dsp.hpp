#ifndef DSP_HPP
#define DSP_HPP

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <tensor.hpp>
#include <unsupported/Eigen/FFT>
#include <vector>

namespace umxcpp
{

const int SUPPORTED_SAMPLE_RATE = 44100;
const int FFT_WINDOW_SIZE = 4096;

const int FFT_HOP_SIZE = 1024; // 25% hop i.e. 75% overlap

// waveform = 2d: (channels, samples)
Eigen::MatrixXf load_audio(std::string filename);

void write_audio_file(const Eigen::MatrixXf &waveform, std::string filename);

// combine magnitude and phase spectrograms into complex
Eigen::Tensor3dXcf polar_to_complex(const Eigen::Tensor3dXf &magnitude,
                                    const Eigen::Tensor3dXf &phase);

Eigen::Tensor3dXcf stft(const Eigen::MatrixXf &audio);
Eigen::MatrixXf istft(const Eigen::Tensor3dXcf &spec);

} // namespace umxcpp

#endif // DSP_HPP
