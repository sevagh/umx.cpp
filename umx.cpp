#include "dsp.hpp"
#include "inference.hpp"
#include "lstm.hpp"
#include "model.hpp"
#include "wiener.hpp"
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

// forward declarations
static std::vector<Eigen::MatrixXf>
shift_inference(struct umxcpp::umx_model &model, Eigen::MatrixXf &full_audio);

static std::vector<Eigen::MatrixXf>
split_inference(struct umxcpp::umx_model &model, Eigen::MatrixXf &full_audio);

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
    std::string model_file = argv[1];

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

    auto ret = load_umx_model(model_file, &model);
    std::cout << "umx_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    std::vector<Eigen::MatrixXf> target_waveforms =
        shift_inference(model, audio);

#pragma omp parallel for
    for (int target = 0; target < 4; ++target)
    {
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

        Eigen::MatrixXf audio_target = target_waveforms[target];
        umxcpp::write_audio_file(audio_target, p_target);
    }
}

static std::vector<Eigen::MatrixXf>
shift_inference(struct umxcpp::umx_model &model, Eigen::MatrixXf &full_audio)
{
    int length = full_audio.cols();

    // first, apply shifts for time invariance
    // we simply only support shift=1, the demucs default
    // shifts (int): if > 0, will shift in time `mix` by a random amount between
    // 0 and 0.5 sec
    //     and apply the oppositve shift to the output. This is repeated
    //     `shifts` time and all predictions are averaged. This effectively
    //     makes the model time equivariant and improves SDR by up to 0.2
    //     points.
    int max_shift =
        (int)(umxcpp::MAX_SHIFT_SECS * umxcpp::SUPPORTED_SAMPLE_RATE);

    int offset = rand() % max_shift;

    // populate padded_full_audio with full_audio starting from
    // max_shift to max_shift + full_audio.cols()
    // incorporate random offset at the same time
    Eigen::MatrixXf shifted_audio =
        Eigen::MatrixXf::Zero(2, length + max_shift - offset);
    shifted_audio.block(0, offset, 2, length) = full_audio;

    std::vector<Eigen::MatrixXf> waveform_outputs =
        split_inference(model, shifted_audio);

    // trim the output to the original length
    // waveform_outputs = waveform_outputs[..., max_shift:max_shift + length]
    std::vector<Eigen::MatrixXf> trimmed_waveform_outputs;

    trimmed_waveform_outputs.push_back(Eigen::MatrixXf(2, length));
    trimmed_waveform_outputs.push_back(Eigen::MatrixXf(2, length));
    trimmed_waveform_outputs.push_back(Eigen::MatrixXf(2, length));
    trimmed_waveform_outputs.push_back(Eigen::MatrixXf(2, length));

    for (int i = 0; i < 4; ++i)
    {
        trimmed_waveform_outputs[i].setZero();
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < length; ++k)
            {
                trimmed_waveform_outputs[i](j, k) =
                    waveform_outputs[i](j, k + offset);
            }
        }
    }

    return trimmed_waveform_outputs;
}

static std::vector<Eigen::MatrixXf>
split_inference(struct umxcpp::umx_model &model, Eigen::MatrixXf &full_audio)
{
    // calculate segment in samples
    int segment_samples =
        (int)(umxcpp::SEGMENT_LEN_SECS * umxcpp::SUPPORTED_SAMPLE_RATE);

    // let's create reusable buffers - LATER
    // struct umxcpp::stft_buffers stft_buf(buffers.segment_samples);
    struct umxcpp::stft_buffers reusable_stft_buf(segment_samples);

    int nb_stft_frames_segment = (segment_samples / umxcpp::FFT_HOP_SIZE + 1);

    int lstm_hidden_size = model.hidden_size / 2;

    std::array<struct umxcpp::lstm_data, 4> streaming_lstm_data = {
        umxcpp::create_lstm_data(lstm_hidden_size, nb_stft_frames_segment),
        umxcpp::create_lstm_data(lstm_hidden_size, nb_stft_frames_segment),
        umxcpp::create_lstm_data(lstm_hidden_size, nb_stft_frames_segment),
        umxcpp::create_lstm_data(lstm_hidden_size, nb_stft_frames_segment)};

    // next, use splits with weighted transition and overlap
    // split (bool): if True, the input will be broken down in 8 seconds
    // extracts
    //     and predictions will be performed individually on each and
    //     concatenated. Useful for model with large memory footprint like
    //     Tasnet.

    int stride_samples = (int)((1 - umxcpp::OVERLAP) * segment_samples);

    int length = full_audio.cols();

    // create an output tensor of zeros for four source waveforms
    std::vector<Eigen::MatrixXf> out;
    out.push_back(Eigen::MatrixXf(2, length));
    out.push_back(Eigen::MatrixXf(2, length));
    out.push_back(Eigen::MatrixXf(2, length));
    out.push_back(Eigen::MatrixXf(2, length));

    for (int i = 0; i < 4; ++i)
    {
        out[i].setZero();
    }

    // create weight tensor
    Eigen::VectorXf weight(segment_samples);
    Eigen::VectorXf sum_weight(length);
    for (int i = 0; i < segment_samples / 2; ++i)
    {
        weight(i) = i + 1;
        weight(segment_samples - i - 1) = i + 1;
        sum_weight(i) = 0.0f;
    }
    weight /= weight.maxCoeff();
    weight = weight.array().pow(umxcpp::TRANSITION_POWER);

    float total_reps = std::ceil(static_cast<float>(length) / stride_samples);
    float per_segment_progress = 1.0f / total_reps;

    std::cout << "Per-segment progress: " << per_segment_progress << std::endl;

    // for loop from 0 to length with stride stride_samples
    for (int offset = 0; offset < length; offset += stride_samples)
    {
        // create a chunk of the padded_full_audio
        int chunk_end = std::min(segment_samples, length - offset);
        Eigen::MatrixXf chunk = full_audio.block(0, offset, 2, chunk_end);
        int chunk_length = chunk.cols();

        std::cout << "2., apply model w/ split, offset: " << offset
                  << ", chunk shape: (" << chunk.rows() << ", " << chunk.cols()
                  << ")" << std::endl;

        // REPLACE THIS WITH UMX INFERENCE!
        std::vector<Eigen::MatrixXf> chunk_out =
            umx_inference(model, chunk, reusable_stft_buf, streaming_lstm_data);

        model.inference_progress += per_segment_progress;

        // add the weighted chunk to the output
        // out[..., offset:offset + segment] += (weight[:chunk_length] *
        // chunk_out).to(mix.device)
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < segment_samples; ++k)
                {
                    if (offset + k >= length)
                    {
                        break;
                    }
                    out[i](j, offset + k) +=
                        weight(k % chunk_length) * chunk_out[i](j, k);
                }
            }
        }

        // sum_weight[offset:offset + segment] +=
        // weight[:chunk_length].to(mix.device)
        for (int k = 0; k < segment_samples; ++k)
        {
            if (offset + k >= length)
            {
                break;
            }
            sum_weight(offset + k) += weight(k % chunk_length);
        }
    }

    assert(sum_weight.minCoeff() > 0);

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < length; ++k)
            {
                out[i](j, k) /= sum_weight[k];
            }
        }
    }

    // now copy the appropriate segment of the output
    // into the output tensor same shape as the input
    std::vector<Eigen::MatrixXf> waveform_outputs;
    waveform_outputs.push_back(Eigen::MatrixXf(2, length));
    waveform_outputs.push_back(Eigen::MatrixXf(2, length));
    waveform_outputs.push_back(Eigen::MatrixXf(2, length));
    waveform_outputs.push_back(Eigen::MatrixXf(2, length));

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < length; ++k)
            {
                waveform_outputs[i](j, k) = out[i](j, k);
            }
        }
    }

    return waveform_outputs;
}
