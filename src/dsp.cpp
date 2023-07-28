#include "dsp.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <memory>
#include <string>
#include <unsupported/Eigen/FFT>
#include <vector>

using namespace nqr;

static constexpr float PI = 3.14159265359F;

Eigen::MatrixXf umxcpp::load_audio(std::string filename)
{
    // load a wav file with libnyquist
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    loader.Load(fileData.get(), filename);

    if (fileData->sampleRate != SUPPORTED_SAMPLE_RATE)
    {
        std::cerr
            << "[ERROR] umx.cpp only supports the following sample rate (Hz): "
            << SUPPORTED_SAMPLE_RATE << std::endl;
        exit(1);
    }

    std::cout << "Input Samples: " << fileData->samples.size() << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;

    if (fileData->channelCount != 2 && fileData->channelCount != 1)
    {
        std::cerr << "[ERROR] umx.cpp only supports mono and stereo audio"
                  << std::endl;
        exit(1);
    }

    // number of samples per channel
    size_t N = fileData->samples.size() / fileData->channelCount;

    // create a struct to hold two float vectors for left and right channels
    Eigen::MatrixXf ret(2, N);

    if (fileData->channelCount == 1)
    {
        // Mono case
        for (size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[i]; // left channel
            ret(1, i) = fileData->samples[i]; // right channel
        }
    }
    else
    {
        // Stereo case
        for (size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[2 * i];     // left channel
            ret(1, i) = fileData->samples[2 * i + 1]; // right channel
        }
    }

    return ret;
}

// write a function to write a StereoWaveform to a wav file
void umxcpp::write_audio_file(const Eigen::MatrixXf &waveform,
                              std::string filename)
{
    // create a struct to hold the audio data
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    // set the sample rate
    fileData->sampleRate = SUPPORTED_SAMPLE_RATE;

    // set the number of channels
    fileData->channelCount = 2;

    // set the number of samples
    fileData->samples.resize(waveform.cols() * 2);

    // write the left channel
    for (size_t i = 0; i < waveform.cols(); ++i)
    {
        fileData->samples[2 * i] = waveform(0, i);
        fileData->samples[2 * i + 1] = waveform(1, i);
    }

    int encoderStatus =
        encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE},
                           fileData.get(), filename);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
}

// forward declaration of inner stft
std::vector<std::vector<std::complex<float>>>
stft_inner(const std::vector<float> &waveform, const std::vector<float> &window,
           int nfft, int hop_size);

std::vector<float>
istft_inner(const std::vector<std::vector<std::complex<float>>> &input,
            const std::vector<float> &window, int nfft, int hop_size);

std::vector<float> hann_window(int window_size)
{
    // create a periodic hann window
    // by generating L+1 points and deleting the last one
    std::size_t N = window_size + 1;

    std::vector<float> window(N);
    auto floatN = (float)(N);

    for (std::size_t n = 0; n < N; ++n)
    {
        window[n] = 0.5F * (1.0F - cosf(2.0F * PI * (float)n / (floatN - 1)));
    }
    // delete the last element
    window.pop_back();
    return window;
}

// reflect padding
void pad_signal(std::vector<float> &signal, int n_fft)
{
    int pad = n_fft / 2;
    std::vector<float> pad_start(signal.begin(), signal.begin() + pad);
    std::vector<float> pad_end(signal.end() - pad, signal.end());
    std::reverse(pad_start.begin(), pad_start.end());
    std::reverse(pad_end.begin(), pad_end.end());
    signal.insert(signal.begin(), pad_start.begin(), pad_start.end());
    signal.insert(signal.end(), pad_end.begin(), pad_end.end());
}

// reflect unpadding
void unpad_signal(std::vector<float> &signal, int n_fft)
{
    int pad = n_fft / 2;
    signal.erase(signal.begin(),
                 signal.begin() + pad); // remove 'pad' elements from the start

    auto it = signal.end() - pad;
    signal.erase(it, signal.end()); // remove 'pad' elements from the end
}

Eigen::Tensor3dXcf umxcpp::polar_to_complex(const Eigen::Tensor3dXf &magnitude,
                                            const Eigen::Tensor3dXf &phase)
{
    // Assert dimensions are the same
    assert(magnitude.dimensions() == phase.dimensions());

    // Get dimensions for convenience
    int dim1 = magnitude.dimension(0);
    int dim2 = magnitude.dimension(1);
    int dim3 = magnitude.dimension(2);

    // Initialize complex spectrogram tensor
    Eigen::Tensor3dXcf complex_spectrogram(dim1, dim2, dim3);

    // Iterate over all indices and apply the transformation
    for (int i = 0; i < dim1; ++i)
    {
        for (int j = 0; j < dim2; ++j)
        {
            for (int k = 0; k < dim3; ++k)
            {
                float mag = magnitude(i, j, k);
                float ph = phase(i, j, k);
                complex_spectrogram(i, j, k) = std::polar(mag, ph);
            }
        }
    }

    return complex_spectrogram;
}

Eigen::Tensor3dXcf umxcpp::stft(const Eigen::MatrixXf &audio)
{
    auto window = hann_window(FFT_WINDOW_SIZE);

    // apply padding equivalent to center padding with center=True
    // in torch.stft:
    // https://pytorch.org/docs/stable/generated/torch.stft.html

    std::vector<float> audio_left(audio.row(0).size());
    Eigen::VectorXf row_vec = audio.row(0);
    std::copy_n(row_vec.data(), row_vec.size(), audio_left.begin());

    std::vector<float> audio_right(audio.row(1).size());
    row_vec = audio.row(1);
    std::copy_n(row_vec.data(), row_vec.size(), audio_right.begin());

    pad_signal(audio_left, FFT_WINDOW_SIZE);
    pad_signal(audio_right, FFT_WINDOW_SIZE);

    auto stft_left =
        stft_inner(audio_left, window, FFT_WINDOW_SIZE, FFT_HOP_SIZE);
    auto stft_right =
        stft_inner(audio_right, window, FFT_WINDOW_SIZE, FFT_HOP_SIZE);

    // get the size of rows and cols
    int rows = stft_left.size();
    int cols = stft_left[0].size();

    Eigen::Tensor3dXcf spec(2, rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            spec(0, i, j) = stft_left[i][j];
            spec(1, i, j) = stft_right[i][j];
        }
    }

    return spec;
}

Eigen::MatrixXf umxcpp::istft(const Eigen::Tensor3dXcf &spec)
{
    auto window = hann_window(FFT_WINDOW_SIZE);

    int rows = spec.dimension(1);
    int cols = spec.dimension(2);

    // Create the nested vectors
    std::vector<std::vector<std::complex<float>>> stft_left(
        rows, std::vector<std::complex<float>>(cols));
    std::vector<std::vector<std::complex<float>>> stft_right(
        rows, std::vector<std::complex<float>>(cols));

    // Populate the nested vectors
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            stft_left[i][j] = spec(0, i, j);
            stft_right[i][j] = spec(1, i, j);
        }
    }

    std::vector<float> chn_left =
        istft_inner(stft_left, window, FFT_WINDOW_SIZE, FFT_HOP_SIZE);
    std::vector<float> chn_right =
        istft_inner(stft_right, window, FFT_WINDOW_SIZE, FFT_HOP_SIZE);

    unpad_signal(chn_left, FFT_WINDOW_SIZE);
    unpad_signal(chn_right, FFT_WINDOW_SIZE);

    Eigen::MatrixXf audio(2, chn_left.size());

    audio.row(0) =
        Eigen::Map<Eigen::MatrixXf>(chn_left.data(), 1, chn_left.size());
    audio.row(1) =
        Eigen::Map<Eigen::MatrixXf>(chn_right.data(), 1, chn_right.size());

    return audio;
}

static Eigen::FFT<float> get_fft_cfg()
{
    Eigen::FFT<float> cfg;
    cfg.SetFlag(Eigen::FFT<float>::Speedy);
    cfg.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    cfg.SetFlag(Eigen::FFT<float>::Unscaled);
    return cfg;
}

std::vector<std::vector<std::complex<float>>>
stft_inner(const std::vector<float> &waveform, const std::vector<float> &window,
           int nfft, int hop_size)
{
    // Check input
    if (waveform.size() < nfft || window.size() != nfft)
    {
        throw std::invalid_argument(
            "Waveform size must be >= nfft, window size must be == nfft.");
    }

    // Output container
    std::vector<std::vector<std::complex<float>>> output;

    // Create an FFT object
    Eigen::FFT<float> cfg = get_fft_cfg();

    // Loop over the waveform with a stride of hop_size
    for (std::size_t start = 0; start <= waveform.size() - nfft;
         start += hop_size)
    {
        // Apply window and run FFT
        std::vector<float> windowed(nfft);
        std::vector<std::complex<float>> spectrum(nfft / 2 + 1);

        for (int i = 0; i < nfft; ++i)
        {
            windowed[i] = waveform[start + i] * window[i];
        }
        cfg.fwd(spectrum, windowed);

        // Add the spectrum to output
        output.push_back(spectrum);
    }

    return output;
}

std::vector<float>
istft_inner(const std::vector<std::vector<std::complex<float>>> &input,
            const std::vector<float> &window, int nfft, int hop_size)
{
    // Check input
    if (input.empty() || input[0].size() != nfft / 2 + 1 ||
        window.size() != nfft)
    {
        throw std::invalid_argument("Input size is not compatible with nfft "
                                    "or window size does not match nfft.");
    }

    // Compute the window normalization factor
    // using librosa window_sumsquare to compute the squared window
    // https://github.com/librosa/librosa/blob/main/librosa/filters.py#L1545

    float win_n = nfft + hop_size * (input.size() - 1);
    std::vector<float> x(win_n, 0.0f);

    for (int i = 0; i < input.size(); ++i)
    {
        auto sample = i * hop_size;
        for (int j = sample; j < std::min((int)win_n, sample + nfft); ++j)
        {
            x[j] += window[j - sample] * window[j - sample];
        }
    }

    // Output container
    std::vector<float> output(win_n, 0.0f);

    // Create an FFT object
    Eigen::FFT<float> cfg = get_fft_cfg();

    // Loop over the input with a stride of (hop_size)
    for (std::size_t start = 0; start < input.size() * hop_size;
         start += hop_size)
    {
        // Run iFFT
        std::vector<float> waveform(nfft);
        cfg.inv(waveform, input[start / hop_size]);

        // Apply window and add to output
        for (int i = 0; i < nfft; ++i)
        {
            // x[start+i] is the sum of squared window values
            // https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py#L613
            // 1e-8f is a small number to avoid division by zero
            output[start + i] += waveform[i] * window[i] * 1.0f / float(nfft) /
                                 (x[start + i] + 1e-8f);
        }
    }

    return output;
}
