// use gtest to test the load_audio_for_kissfft function

#include "dsp.hpp"
#include <gtest/gtest.h>
#include <random>

#define NEAR_TOLERANCE 1e-4

// write a basic test case for a mono file
TEST(LoadAudioForKissfft, LoadMonoAudio)
{
    // load a wav file with libnyquist
    std::string filename = "../test/data/gspi_mono.wav";
    Eigen::MatrixXf ret = umxcpp::load_audio(filename);

    // check the number of samples
    EXPECT_EQ(ret.cols(), 262144);

    // check the first and last samples
    EXPECT_EQ(ret(0, 0), ret(1, 0));
    EXPECT_EQ(ret(0, 262143), ret(1, 262143));
}

// write a basic test case for a stereo file
TEST(LoadAudioForKissfft, LoadStereoAudio)
{
    // load a wav file with libnyquist
    std::string filename = "../test/data/gspi_stereo.wav";

    Eigen::MatrixXf ret = umxcpp::load_audio(filename);

    // check the number of samples
    EXPECT_EQ(ret.cols(), 262144);

    // check the first and last samples
    EXPECT_EQ(ret(0, 0), ret(1, 0));
    EXPECT_EQ(ret(0, 262143), ret(1, 262143));
}

// write a basic test case for the stft function
TEST(DSP_STFT, STFTRoundtripRandWaveform)
{
    Eigen::MatrixXf audio_in(2, 4096);

    // populate the audio_in with some random data
    // between -1 and 1
    for (size_t i = 0; i < 4096; ++i)
    {
        audio_in(0, i) = (float)rand() / (float)RAND_MAX;
        audio_in(1, i) = (float)rand() / (float)RAND_MAX;
    }

    // compute the stft
    Eigen::Tensor3dXcf spec = umxcpp::stft(audio_in);

    // check the number of frequency bins per frames, first and last
    auto n_frames = spec.dimension(1);

    std::cout << "n_frames: " << n_frames << std::endl;

    EXPECT_EQ(spec.dimension(2), 2049);

    Eigen::MatrixXf audio_out = umxcpp::istft(spec);

    EXPECT_EQ(audio_in.rows(), audio_out.rows());
    EXPECT_EQ(audio_in.cols(), audio_out.cols());

    for (size_t i = 0; i < audio_in.cols(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in(0, i), audio_out(0, i), NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in(1, i), audio_out(1, i), NEAR_TOLERANCE);
    }
}

// write a basic test case for the stft function
// with real gspi.wav
TEST(DSP_STFT, STFTRoundtripGlockenspiel)
{
    Eigen::MatrixXf audio_in =
        umxcpp::load_audio("../test/data/gspi_mono.wav");

    // compute the stft
    Eigen::Tensor3dXcf spec = umxcpp::stft(audio_in);

    // check the number of frequency bins per frames, first and last
    auto n_frames = spec.dimension(1);

    std::cout << "n_frames: " << n_frames << std::endl;

    EXPECT_EQ(spec.dimension(2), 2049);

    Eigen::MatrixXf audio_out = umxcpp::istft(spec);

    EXPECT_EQ(audio_in.rows(), audio_out.rows());
    EXPECT_EQ(audio_in.cols(), audio_out.cols());

    for (size_t i = 0; i < audio_in.cols(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in(0, i), audio_out(0, i), NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in(1, i), audio_out(1, i), NEAR_TOLERANCE);
    }
}

// write a test for the magnitude and phase functions
// and test a roundtrip with the combine function
TEST(DSP_STFT, MagnitudePhaseCombineMono)
{
    Eigen::MatrixXf audio_in =
        umxcpp::load_audio("../test/data/gspi_mono.wav");

    // compute the stft
    Eigen::Tensor3dXcf spec = umxcpp::stft(audio_in);

    // compute the magnitude and phase
    Eigen::Tensor3dXf mag = spec.unaryExpr(
        [](const std::complex<float> &c) { return std::abs(c); });

    Eigen::Tensor3dXf phase = spec.unaryExpr(
        [](const std::complex<float> &c) { return std::arg(c); });

    // print each dimension of spec
    std::cout << "spec.dimensions(): ";
    for (size_t i = 0; i < spec.dimensions().size(); ++i)
    {
        std::cout << spec.dimensions()[i] << " ";
    }
    std::cout << std::endl;

    // ensure all magnitude are positive
    for (size_t chan = 0; chan < mag.dimension(0); ++chan)
    {
        for (size_t i = 0; i < mag.dimension(1); ++i)
        {
            for (size_t j = 0; j < mag.dimension(2); ++j)
            {
                EXPECT_GE(mag(chan, i, j), 0.0);
            }
        }
    }

    // combine the magnitude and phase
    Eigen::Tensor3dXcf spec2 = umxcpp::polar_to_complex(mag, phase);

    // ensure spec and spec2 are the same
    // first check their sizes
    EXPECT_EQ(spec.dimensions().size(), spec2.dimensions().size());
    for (size_t i = 0; i < spec.dimensions().size(); ++i)
    {
        EXPECT_EQ(spec.dimensions()[i], spec2.dimensions()[i]);
    }

    for (size_t chan = 0; chan < spec.dimension(0); ++chan)
    {
        for (size_t i = 0; i < spec.dimension(1); ++i)
        {
            for (size_t j = 0; j < spec.dimension(2); ++j)
            {
                EXPECT_NEAR(std::real(spec(chan, i, j)), std::real(spec2(chan, i, j)),
                            NEAR_TOLERANCE);
                EXPECT_NEAR(std::imag(spec(chan, i, j)), std::imag(spec2(chan, i, j)),
                            NEAR_TOLERANCE);
            }
        }
    }

    // compute the istft
    Eigen::MatrixXf audio_out = umxcpp::istft(spec2);

    EXPECT_EQ(audio_in.rows(), audio_out.rows());
    EXPECT_EQ(audio_in.cols(), audio_out.cols());

    for (size_t i = 0; i < audio_in.cols(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in(0, i), audio_out(0, i), NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in(1, i), audio_out(1, i), NEAR_TOLERANCE);
    }
}

// write a test for the magnitude and phase functions
// and test a roundtrip with the combine function
TEST(DSP_STFT, MagnitudePhaseCombineStereo)
{
    Eigen::MatrixXf audio_in =
        umxcpp::load_audio("../test/data/gspi_stereo.wav");

    // compute the stft
    Eigen::Tensor3dXcf spec = umxcpp::stft(audio_in);

    // compute the magnitude and phase
    Eigen::Tensor3dXf mag = spec.unaryExpr(
        [](const std::complex<float> &c) { return std::abs(c); });

    Eigen::Tensor3dXf phase = spec.unaryExpr(
        [](const std::complex<float> &c) { return std::arg(c); });

    // ensure all magnitude are positive
    for (size_t chan = 0; chan < mag.dimension(0); ++chan)
    {
        for (size_t i = 0; i < mag.dimension(1); ++i)
        {
            for (size_t j = 0; j < mag.dimension(2); ++j)
            {
                EXPECT_GE(mag(chan, i, j), 0.0);
            }
        }
    }

    // combine the magnitude and phase
    Eigen::Tensor3dXcf spec2 = umxcpp::polar_to_complex(mag, phase);

    // ensure spec and spec2 are the same
    // first check their sizes
    EXPECT_EQ(spec.dimensions().size(), spec2.dimensions().size());
    for (size_t i = 0; i < spec.dimensions().size(); ++i)
    {
        EXPECT_EQ(spec.dimensions()[i], spec2.dimensions()[i]);
    }

    for (size_t chan = 0; chan < spec.dimension(0); ++chan)
    {
        for (size_t i = 0; i < spec.dimension(1); ++i)
        {
            for (size_t j = 0; j < spec.dimension(2); ++j)
            {
                EXPECT_NEAR(std::real(spec(chan, i, j)), std::real(spec2(chan, i, j)),
                            NEAR_TOLERANCE);
                EXPECT_NEAR(std::imag(spec(chan, i, j)), std::imag(spec2(chan, i, j)),
                            NEAR_TOLERANCE);
            }
        }
    }

    // compute the istft
    Eigen::MatrixXf audio_out = umxcpp::istft(spec2);

    EXPECT_EQ(audio_in.rows(), audio_out.rows());
    EXPECT_EQ(audio_in.cols(), audio_out.cols());

    for (size_t i = 0; i < audio_in.cols(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in(0, i), audio_out(0, i), NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in(1, i), audio_out(1, i), NEAR_TOLERANCE);
    }
}
