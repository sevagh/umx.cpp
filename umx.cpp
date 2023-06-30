#include "dsp.hpp"
#include "lstm.hpp"
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

    StereoWaveform audio = load_audio(wav_file);

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
    StereoSpectrogramC spectrogram = stft(audio);

    std::cout << "spec shape: (incl 2 chan) " << spectrogram.left.size()
              << " x " << spectrogram.left[0].size() << std::endl;

    std::cout << "Computing STFT magnitude" << std::endl;
    // now let's get a stereo magnitude spectrogram
    StereoSpectrogramR mix_mag = magnitude(spectrogram);

    std::cout << "Computing STFT phase" << std::endl;
    StereoSpectrogramR mix_phase = phase(spectrogram);

    // apply umx inference to the magnitude spectrogram
    // first create a ggml_tensor for the input

    int hidden_size = model.hidden_size;

    // input shape is (nb_frames*nb_samples, nb_channels*nb_bins) i.e. 2049*2
    int nb_bins = 2049;

    int nb_frames = mix_mag.left.size();
    int nb_real_bins = mix_mag.left[0].size();

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
            x(i, j) = mix_mag.left[i][j];
            // fill second half of 2974/2 bins from right
            x(i, j + nb_bins_cropped) = mix_mag.right[i][j];
        }
    }

    std::cout << "Running inference with Eigen matrices" << std::endl;

    // clone input mix mag x to operate on targets x_{0,1,2,3}
    Eigen::MatrixXf x_inputs[4];

#pragma omp parallel for
    for (int target = 0; target < 4; ++target)
    {
        x_inputs[target] = x;
// opportunistically apply input scaling and mean

// apply formula x = x*input_scale + input_mean
#pragma omp parallel for
        for (int i = 0; i < x_inputs[target].rows(); i++)
        {
            x_inputs[target].row(i) = x_inputs[target].row(i).array() *
                                          model.input_scale[target].array() +
                                      model.input_mean[target].array();
        }
    }

    // create pointer to a Eigen::MatrixXf to modify in the for loop
    // there are classes in Eigen for this

#pragma omp parallel for
    for (int target = 0; target < 4; ++target)
    {
        // y = x A^T + b
        // A = weights = (out_features, in_features)
        // A^T = A transpose = (in_features, out_features)
        x_inputs[target] = x_inputs[target] * model.fc1_w[target];

// batchnorm1d calculation
// y=(x-E[x])/(sqrt(Var[x]+ϵ) * gamma + Beta
#pragma omp parallel for
        for (int i = 0; i < x_inputs[target].rows(); i++)
        {
            x_inputs[target].row(i) =
                (((x_inputs[target].row(i).array() -
                   model.bn1_rm[target].array()) /
                  (model.bn1_rv[target].array() + 1e-5).sqrt()) *
                     model.bn1_w[target].array() +
                 model.bn1_b[target].array())
                    .tanh();
        }

        // now lstm time
        int lstm_hidden_size = hidden_size / 2;

        // umx_lstm_forward applies bidirectional 3-layer lstm using a
        // LSTMCell-like approach
        // https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

        auto lstm_out_0 = umxcpp::umx_lstm_forward(
            model, target, x_inputs[target], lstm_hidden_size);

        // now the concat trick from umx for the skip conn
        //    # apply 3-layers of stacked LSTM
        //    lstm_out = self.lstm(x)
        //    # lstm skip connection
        //    x = torch.cat([x, lstm_out[0]], -1)
        // concat the lstm_out with the input x
        Eigen::MatrixXf x_inputs_target_concat(x_inputs[target].rows(),
                                               x_inputs[target].cols() +
                                                   lstm_out_0.cols());
        x_inputs_target_concat.leftCols(x_inputs[target].cols()) =
            x_inputs[target];
        x_inputs_target_concat.rightCols(lstm_out_0.cols()) = lstm_out_0;

        x_inputs[target] = x_inputs_target_concat;

        // now time for fc2
        x_inputs[target] = x_inputs[target] * model.fc2_w[target];

// batchnorm1d calculation
// y=(x-E[x])/(sqrt(Var[x]+ϵ) * gamma + Beta
#pragma omp parallel for
        for (int i = 0; i < x_inputs[target].rows(); i++)
        {
            x_inputs[target].row(i) =
                (((x_inputs[target].row(i).array() -
                   model.bn2_rm[target].array()) /
                  (model.bn2_rv[target].array() + 1e-5).sqrt()) *
                     model.bn2_w[target].array() +
                 model.bn2_b[target].array())
                    .cwiseMax(0);
        }

        x_inputs[target] = x_inputs[target] * model.fc3_w[target];

// batchnorm1d calculation
// y=(x-E[x])/(sqrt(Var[x]+ϵ) * gamma + Beta
#pragma omp parallel for
        for (int i = 0; i < x_inputs[target].rows(); i++)
        {
            x_inputs[target].row(i) =
                ((x_inputs[target].row(i).array() -
                  model.bn3_rm[target].array()) /
                 (model.bn3_rv[target].array() + 1e-5).sqrt()) *
                    model.bn3_w[target].array() +
                model.bn3_b[target].array();
        }

// now output scaling
// apply formula x = x*output_scale + output_mean
#pragma omp parallel for
        for (int i = 0; i < x_inputs[target].rows(); i++)
        {
            x_inputs[target].row(i) = (x_inputs[target].row(i).array() *
                                           model.output_scale[target].array() +
                                       model.output_mean[target].array())
                                          .cwiseMax(0);
        }

        // print min and max elements of x_inputs[target]
        std::cout << "POST-RELU-FINAL x_inputs[target] min: "
                  << x_inputs[target].minCoeff()
                  << " x_inputs[target] max: " << x_inputs[target].maxCoeff()
                  << std::endl;

        // copy mix-mag
        StereoSpectrogramR mix_mag_target(mix_mag);

// element-wise multiplication, taking into account the stacked outputs of the
// neural network
#pragma omp parallel for
        for (std::size_t i = 0; i < mix_mag.left.size(); i++)
        {
#pragma omp parallel for
            for (std::size_t j = 0; j < mix_mag.left[0].size(); j++)
            {
                mix_mag_target.left[i][j] *= x_inputs[target](i, j);
                mix_mag_target.right[i][j] *=
                    x_inputs[target](i, j + mix_mag.left[0].size());
            }
        }

        // now let's get a stereo waveform back first with phase
        StereoSpectrogramC mix_complex_target =
            combine(mix_mag_target, mix_phase);

        StereoWaveform audio_target = istft(mix_complex_target);

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
