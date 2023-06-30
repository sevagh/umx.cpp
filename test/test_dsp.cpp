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
    umxcpp::StereoWaveform ret = umxcpp::load_audio(filename);

    // check the number of samples
    EXPECT_EQ(ret.left.size(), 262144);
    EXPECT_EQ(ret.right.size(), 262144);

    // check the first and last samples
    EXPECT_EQ(ret.left[0], ret.right[0]);
    EXPECT_EQ(ret.left[262143], ret.right[262143]);
}

// write a basic test case for a stereo file
TEST(LoadAudioForKissfft, LoadStereoAudio)
{
    // load a wav file with libnyquist
    std::string filename = "../test/data/gspi_stereo.wav";
    umxcpp::StereoWaveform ret = umxcpp::load_audio(filename);

    // check the number of samples
    EXPECT_EQ(ret.left.size(), 262144);
    EXPECT_EQ(ret.right.size(), 262144);

    // check the first and last samples
    EXPECT_EQ(ret.left[0], ret.right[0]);
    EXPECT_EQ(ret.left[262143], ret.right[262143]);
}

// write a basic test case for the stft function
TEST(DSP_STFT, STFTRoundtripRandWaveform)
{
    umxcpp::StereoWaveform audio_in;

    audio_in.left.resize(4096);
    audio_in.right.resize(4096);

    // populate the audio_in with some random data
    // between -1 and 1
    for (size_t i = 0; i < 4096; ++i)
    {
        audio_in.left[i] = (float)rand() / (float)RAND_MAX;
        audio_in.right[i] = (float)rand() / (float)RAND_MAX;
    }

    // compute the stft
    umxcpp::StereoSpectrogramC spec = umxcpp::stft(audio_in);

    // check the number of frequency bins per frames, first and last
    auto n_frames = spec.left.size();
    EXPECT_EQ(n_frames, spec.right.size());

    std::cout << "n_frames: " << n_frames << std::endl;

    EXPECT_EQ(spec.left[0].size(), 2049);
    EXPECT_EQ(spec.right[0].size(), 2049);
    EXPECT_EQ(spec.left[n_frames - 1].size(), 2049);
    EXPECT_EQ(spec.right[n_frames - 1].size(), 2049);

    umxcpp::StereoWaveform audio_out = umxcpp::istft(spec);

    EXPECT_EQ(audio_in.left.size(), audio_out.left.size());
    EXPECT_EQ(audio_in.right.size(), audio_out.right.size());

    for (size_t i = 0; i < audio_in.left.size(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in.left[i], audio_out.left[i], NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in.right[i], audio_out.right[i], NEAR_TOLERANCE);
    }
}

// write a basic test case for the stft function
// with real gspi.wav
TEST(DSP_STFT, STFTRoundtripGlockenspiel)
{
    umxcpp::StereoWaveform audio_in =
        umxcpp::load_audio("../test/data/gspi_mono.wav");

    // compute the stft
    umxcpp::StereoSpectrogramC spec = umxcpp::stft(audio_in);

    // check the number of frequency bins per frames, first and last
    auto n_frames = spec.left.size();
    EXPECT_EQ(n_frames, spec.right.size());

    std::cout << "n_frames: " << n_frames << std::endl;

    EXPECT_EQ(spec.left[0].size(), 2049);
    EXPECT_EQ(spec.right[0].size(), 2049);
    EXPECT_EQ(spec.left[n_frames - 1].size(), 2049);
    EXPECT_EQ(spec.right[n_frames - 1].size(), 2049);

    umxcpp::StereoWaveform audio_out = umxcpp::istft(spec);

    EXPECT_EQ(audio_in.left.size(), audio_out.left.size());
    EXPECT_EQ(audio_in.right.size(), audio_out.right.size());

    for (size_t i = 0; i < audio_in.left.size(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in.left[i], audio_out.left[i], NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in.right[i], audio_out.right[i], NEAR_TOLERANCE);
    }
}

// write a test for the magnitude and phase functions
// and test a roundtrip with the combine function
TEST(DSP_STFT, MagnitudePhaseCombine)
{
    umxcpp::StereoWaveform audio_in =
        umxcpp::load_audio("../test/data/gspi_mono.wav");

    // compute the stft
    umxcpp::StereoSpectrogramC spec = umxcpp::stft(audio_in);

    // compute the magnitude and phase
    umxcpp::StereoSpectrogramR mag = umxcpp::magnitude(spec);
    umxcpp::StereoSpectrogramR phase = umxcpp::phase(spec);

    // ensure all magnitude are positive
    for (size_t i = 0; i < mag.left.size(); ++i)
    {
        for (size_t j = 0; j < mag.left[i].size(); ++j)
        {
            EXPECT_GE(mag.left[i][j], 0.0);
            EXPECT_GE(mag.right[i][j], 0.0);
        }
    }

    // combine the magnitude and phase
    umxcpp::StereoSpectrogramC spec2 = umxcpp::combine(mag, phase);

    // ensure spec and spec2 are the same
    // first check their sizes
    EXPECT_EQ(spec.left.size(), spec2.left.size());

    for (size_t i = 0; i < spec.left.size(); ++i)
    {
        for (size_t j = 0; j < spec.left[i].size(); ++j)
        {
            EXPECT_NEAR(std::real(spec.left[i][j]), std::real(spec2.left[i][j]),
                        NEAR_TOLERANCE);
            EXPECT_NEAR(std::imag(spec.left[i][j]), std::imag(spec2.left[i][j]),
                        NEAR_TOLERANCE);
            EXPECT_NEAR(std::real(spec.right[i][j]),
                        std::real(spec2.right[i][j]), NEAR_TOLERANCE);
            EXPECT_NEAR(std::imag(spec.right[i][j]),
                        std::imag(spec2.right[i][j]), NEAR_TOLERANCE);
        }
    }

    // compute the istft
    umxcpp::StereoWaveform audio_out = umxcpp::istft(spec2);

    EXPECT_EQ(audio_in.left.size(), audio_out.left.size());
    EXPECT_EQ(audio_in.right.size(), audio_out.right.size());

    for (size_t i = 0; i < audio_in.left.size(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in.left[i], audio_out.left[i], NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in.right[i], audio_out.right[i], NEAR_TOLERANCE);
    }
}

// write a test for the magnitude and phase functions
// and test a roundtrip with the combine function
TEST(DSP_STFT, MagnitudePhaseCombineStereo)
{
    umxcpp::StereoWaveform audio_in =
        umxcpp::load_audio("../test/data/gspi_stereo.wav");

    // compute the stft
    umxcpp::StereoSpectrogramC spec = umxcpp::stft(audio_in);

    // compute the magnitude and phase
    umxcpp::StereoSpectrogramR mag = umxcpp::magnitude(spec);
    umxcpp::StereoSpectrogramR phase = umxcpp::phase(spec);

    // ensure all magnitude are positive
    for (size_t i = 0; i < mag.left.size(); ++i)
    {
        for (size_t j = 0; j < mag.left[i].size(); ++j)
        {
            EXPECT_GE(mag.left[i][j], 0.0);
            EXPECT_GE(mag.right[i][j], 0.0);
        }
    }

    // combine the magnitude and phase
    umxcpp::StereoSpectrogramC spec2 = umxcpp::combine(mag, phase);

    // ensure spec and spec2 are the same
    // first check their sizes
    EXPECT_EQ(spec.left.size(), spec2.left.size());

    for (size_t i = 0; i < spec.left.size(); ++i)
    {
        for (size_t j = 0; j < spec.left[i].size(); ++j)
        {
            // rewrite above for std::complex<float>
            EXPECT_NEAR(std::real(spec.left[i][j]), std::real(spec2.left[i][j]),
                        NEAR_TOLERANCE);
            EXPECT_NEAR(std::imag(spec.left[i][j]), std::imag(spec2.left[i][j]),
                        NEAR_TOLERANCE);
            EXPECT_NEAR(std::real(spec.right[i][j]),
                        std::real(spec2.right[i][j]), NEAR_TOLERANCE);
            EXPECT_NEAR(std::imag(spec.right[i][j]),
                        std::imag(spec2.right[i][j]), NEAR_TOLERANCE);
        }
    }

    // compute the istft
    umxcpp::StereoWaveform audio_out = umxcpp::istft(spec2);

    EXPECT_EQ(audio_in.left.size(), audio_out.left.size());
    EXPECT_EQ(audio_in.right.size(), audio_out.right.size());

    for (size_t i = 0; i < audio_in.left.size(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in.left[i], audio_out.left[i], NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in.right[i], audio_out.right[i], NEAR_TOLERANCE);
    }
}
