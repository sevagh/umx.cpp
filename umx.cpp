#include "dsp.hpp"
#include "lstm.hpp"
#include "wiener.hpp"
#include "model.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unsupported/Eigen/FFT>
#include <vector>

using namespace umxcpp;

int main(int argc, const char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model dir> <wav file> <out dir>"
                  << std::endl;
        exit(1);
    }

    std::cout << "umx.cpp Main driver program" << std::endl;

    // load model passed as argument
    std::string model_dir = argv[1];

    // load audio passed as argument
    std::string wav_file = argv[2];

    // output dir passed as argument
    std::string out_dir = argv[3];

    // init parallelism for eigen
    Eigen::initParallel();

    // set eigen nb threads to physical cores minus 1
    // discover number of physical cores through C++ stdlib
    // https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
    int nb_cores = std::thread::hardware_concurrency();
    std::cout << "Number of physical cores: " << nb_cores << std::endl;
    Eigen::setNbThreads(nb_cores - 1);

    Eigen::MatrixXf audio = load_audio(wav_file);

    // initialize a struct umx_model
    struct umx_model model
    {
    };

    auto ret = load_umx_model(model_dir, &model);
    std::cout << "umx_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    // let's get a stereo complex spectrogram first
    std::cout << "Computing STFT" << std::endl;
    Eigen::Tensor3dXcf spectrogram = stft(audio);

    std::array<Eigen::Tensor3dXcf, 4> target_spectrograms;

    std::cout << "spec shape: (" << spectrogram.dimensions()[0] << ", "
        << spectrogram.dimensions()[1] << ", " << spectrogram.dimensions()[2] << ")"
        << std::endl;

    std::cout << "Computing STFT magnitude" << std::endl;
    // now let's get a stereo magnitude spectrogram
    Eigen::Tensor3dXf mix_mag = spectrogram.abs();

    std::cout << "Computing STFT phase" << std::endl;
    Eigen::Tensor3dXf mix_phase = spectrogram.unaryExpr(
        [](const std::complex<float> &c) { return std::arg(c); });

    // apply umx inference to the magnitude spectrogram
    // first create a ggml_tensor for the input

    int hidden_size = model.hidden_size;

    // input shape is (nb_frames*nb_samples, nb_channels*nb_bins) i.e. 2049*2
    int nb_bins = 2049;

    int nb_frames = mix_mag.dimension(1);
    int nb_real_bins = mix_mag.dimension(2);

    assert(nb_real_bins == nb_bins);

    Eigen::MatrixXf x(nb_frames, 2974);

    // TODO: replace Eigen with ggml (or ggml with Eigen)
    // https://github.com/ggerganov/ggml/discussions/297
    // struct ggml_tensor *x_ggml = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32,
    // 2974, nb_frames);

    int nb_bins_cropped = 2974 / 2;

#pragma omp parallel for
    for (int i = 0; i < nb_frames; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < nb_bins_cropped; j++)
        {
            // interleave fft frames from each channel
            // fill first half of 2974/2 bins from left
            x(i, j) = mix_mag(0, i, j);
            // fill second half of 2974/2 bins from right
            x(i, j + nb_bins_cropped) = mix_mag(1, i, j);
        }
    }

    std::cout << "Running inference with Eigen matrices" << std::endl;

    std::array<Eigen::MatrixXf, 4> x_outputs =
        umx_inference(&model, x, hidden_size);

#pragma omp parallel for
    for (int target = 0; target < 4; ++target) {
        std::cout << "POST-RELU-FINAL x_outputs[target] min: "
                  << x_outputs[target].minCoeff()
                  << " x_inputs[target] max: " << x_outputs[target].maxCoeff()
                  << std::endl;

        // copy mix-mag
        Eigen::Tensor3dXf mix_mag_target(mix_mag);

        // element-wise multiplication, taking into account the stacked outputs of the
        // neural network
#pragma omp parallel for
        for (std::size_t i = 0; i < mix_mag.dimension(1); i++)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < mix_mag.dimension(2); j++)
            {
                mix_mag_target(0, i, j) *= x_outputs[target](i, j);
                mix_mag_target(1, i, j) *=
                    x_outputs[target](i, j + mix_mag.dimension(2));
            }
        }

        // now let's get a stereo waveform back first with phase
        // initial estimate
        target_spectrograms[target] =
            polar_to_complex(mix_mag_target, mix_phase);
    }

#pragma omp parallel for
    for (int target = 0; target < 4; target++) {
        std::cout << "Now running wiener filter" << std::endl;
        auto refined_spectrograms = wiener_filter(
            spectrogram, target_spectrograms);

        // TODO: use refined_spectrograms here
        Eigen::MatrixXf audio_target = istft(refined_spectrograms[target]);

        // now write the 4 audio waveforms to files in the output dir
        // using libnyquist
        // join out_dir with "/target_0.wav"
        // using std::filesystem::path;

        std::filesystem::path p = out_dir;
        // make sure the directory exists
        std::filesystem::create_directories(p);

        auto p_target = p / "target_0.wav";
        // generate p_target = p / "target_{target}.wav"
        p_target.replace_filename("target_" + std::to_string(target) + ".wav");

        std::cout << "Writing wav file " << p_target << " to " << out_dir
                  << std::endl;

        umxcpp::write_audio_file(audio_target, p_target);
    }
}
