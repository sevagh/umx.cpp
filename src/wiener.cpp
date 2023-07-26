#include "wiener.hpp"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorChipping.h>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

// forward declaration of helper functions
// Function signature of expectation_maximization
static Eigen::Tensor4dXcf
expectation_maximization(
    const Eigen::Tensor4dXcf &y_in,
    const Eigen::Tensor3dXcf &x,
    int iterations);

// Wiener filter function
std::array<Eigen::Tensor3dXcf, 4> umxcpp::wiener_filter(
    const Eigen::Tensor3dXcf &mix_stft,
    const std::array<Eigen::Tensor3dXcf, 4> &targets_spectrograms
) {
       // Define a Tensor of 1s of type float
    Eigen::Tensor3dXf onesTensor = Eigen::Tensor3dXf(mix_stft.dimension(0), mix_stft.dimension(1), mix_stft.dimension(2));
    onesTensor.setConstant(1.0f);

    // Compute absolute values of mix_stft
    Eigen::Tensor3dXf sqrtAbs = mix_stft.abs().unaryExpr([](std::complex<float> x) {return std::sqrt(std::abs(x));});

    // Find max value of tensor
    float maxCoeff = *std::max_element(sqrtAbs.data(), sqrtAbs.data() + sqrtAbs.size());

    // Use the tensor-wise operations sqrt and max
    Eigen::Tensor3dXf maxAbs = onesTensor.cwiseMax((float)(maxCoeff / WIENER_SCALE_FACTOR));

    // Scale down estimates
    Eigen::Tensor3dXcf mix_stft_scaled = mix_stft / maxAbs.cast<std::complex<float>>();

    // Make a copy of targets_spectrograms
    std::array<Eigen::Tensor3dXcf, 4> targets_spectrograms_copy = targets_spectrograms;

    for (auto& y : targets_spectrograms_copy) {
        y = y / maxAbs.cast<std::complex<float>>();
    }

    // combine array of targets_spectrograms into a single tensor
    // last dimension is the targets, so 4 total
    Eigen::Tensor4dXcf y_single = Eigen::Tensor4dXcf(mix_stft_scaled.dimension(0), mix_stft_scaled.dimension(1), mix_stft_scaled.dimension(2), 4);
    for (int target = 0; target < 4; ++target) {
        y_single.chip<3>(target) = targets_spectrograms_copy[target];
    }

    y_single = expectation_maximization(y_single, mix_stft_scaled, 1);

    // Scale up estimates
    for (auto& y : targets_spectrograms_copy) {
        y = y * maxAbs.cast<std::complex<float>>();
    }

    return targets_spectrograms_copy;

}

// Function definitions for the helper functions
static Eigen::Tensor4dXcf _covariance(const Eigen::Tensor4dXcf& y) {
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(3, 2)};
    return y.contract(y.conjugate(), product_dims).mean(Eigen::array<int, 1>{0});
}

static Eigen::Tensor4dXcf _invert(const Eigen::Tensor4dXcf& Cxx) {
    // Use eigenvalue decomposition to invert the covariance matrix
    int n1 = Cxx.dimension(0);
    int n2 = Cxx.dimension(1);
    int n3 = Cxx.dimension(2);
    int n4 = Cxx.dimension(3);

    Eigen::Tensor4dXcf invCxx(n1, n2, n3, n4);
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            Eigen::MatrixXcf mat(n3, n4);
            for (int k = 0; k < n3; ++k) {
                for (int l = 0; l < n4; ++l) {
                    mat(k, l) = Cxx(i, j, k, l);
                }
            }
            Eigen::MatrixXcf invMat = mat.inverse();
            for (int k = 0; k < n3; ++k) {
                for (int l = 0; l < n4; ++l) {
                    invCxx(i, j, k, l) = invMat(k, l);
                }
            }
        }
    }
    return invCxx;
}

static Eigen::Tensor4dXcf _apply_filter(
    const Eigen::Tensor4dXcf& y,
    const Eigen::Tensor3dXcf& x,
    const Eigen::Tensor4dXcf& gain,
    int nb_sources,
    int nb_channels
) {
    Eigen::Tensor4dXcf y_out = y;
    for (int j = 0; j < nb_sources; ++j) {
        for (int i = 0; i < nb_channels; ++i) {
            auto y_slice = y_out.chip<4>(j);
            auto gain_slice = gain.chip<3>(i);
            y_slice += gain_slice * x.chip<2>(i);
        }
    }
    return y_out;
}

// Function signature of expectation_maximization
static Eigen::Tensor4dXcf
expectation_maximization(
    const Eigen::Tensor4dXcf &y_in,
    const Eigen::Tensor3dXcf &x,
    int iterations
) {
    // dimensions
    int nb_frames = x.dimension(0);
    int nb_bins = x.dimension(1);
    int nb_channels = x.dimension(2);
    int nb_sources = y_in.dimension(4);

    Eigen::Tensor4dXcf y = y_in;  // create a copy of y to update it in-place

    // Creating regularization tensor
    Eigen::MatrixXcf eye = Eigen::MatrixXcf::Identity(nb_channels, nb_channels);
    Eigen::MatrixXcf zeros = Eigen::MatrixXcf::Zero(nb_channels, nb_channels);

    Eigen::Tensor4dXcf regularization(1, nb_bins, nb_channels, nb_channels * 2);

    // manually perform the concatenation
    for (int i = 0; i < nb_channels; ++i) {
        for (int j = 0; j < nb_channels; ++j) {
            regularization(0, 0, i, j) = eye(i, j);                 // fill with identity
            regularization(0, 0, i, j + nb_channels) = zeros(i, j); // fill with zeros
        }
    }

    // expand into 4D tensor
    Eigen::Tensor4dXcf expanded_regularization(nb_frames, nb_bins, nb_channels, nb_channels * 2);
    for (int i = 0; i < nb_frames; ++i) {
        expanded_regularization.chip<0>(i) = regularization;
    }

    // multiply by sqrt of eps
    expanded_regularization = sqrt(umxcpp::WIENER_EPS) * expanded_regularization;

    // reassign
    regularization = expanded_regularization;


    // allocate the spatial covariance matrices
    std::vector<Eigen::Tensor4dXcf> R(nb_sources, Eigen::Tensor4dXcf(nb_bins, nb_channels, nb_channels, 2));
    Eigen::Tensor3dXf weight(nb_bins, nb_channels, 2);
    Eigen::Tensor3dXf v(nb_frames, nb_bins, nb_sources);

    for (int it = 0; it < iterations; ++it) {
        // y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
        //     initial estimates for the sources
        // # update the PSD as the average spectrogram over channels
        // v = torch.mean(torch.abs(y[..., 0, :]) ** 2 + torch.abs(y[..., 1, :]) ** 2, dim=-2)
        // Reorder the dimensions so that 'channels' is last
        Eigen::array<int, 4> shuffle_dims = {1, 2, 3, 0};
        auto y_shuffled = y.shuffle(shuffle_dims);

        // Compute the absolute squared values of each complex number
        auto y_abs_squared = y_shuffled.abs().square();

        // Compute the mean along the last dimension (channels)
        Eigen::array<int, 1> mean_dims = {3};
        auto v = y_abs_squared.mean(mean_dims);

        // update spatial covariance matrices
        for (int j = 0; j < nb_sources; ++j) {
            auto yj = y.chip<3>(j);
            R[j] = _covariance(yj) + regularization;
        }

        // compute mix covariance matrix
        Eigen::Tensor4dXcf Cxx(nb_bins, nb_channels, nb_channels, nb_sources);
        Cxx.setConstant(std::complex<float>{0.0f, 0.0f});

        for (int j = 0; j < nb_sources; ++j) {
            auto vj = v.chip<2>(j);
            Eigen::Tensor4dXf vjtmp = vj.reshape(Eigen::array<int, 4>{nb_bins, 1, 1, nb_sources});

            // convert real tensor vjtmp to equivalent complex with 1.0f as the imaginary part
            Eigen::Tensor4dXcf vjtmp_complex = vjtmp.unaryExpr([](float x) {return std::complex<float>(x, 1.0f);});

            Cxx += vjtmp_complex * R[j];
        }

        // invert it
        Cxx = _invert(Cxx);

        for (int j = 0; j < nb_sources; ++j) {
            auto vj = v.chip<2>(j);

            Eigen::Tensor4dXf vjtmp = vj.reshape(Eigen::array<int, 4>{nb_bins, 1, 1, nb_sources});

            // convert real tensor vjtmp to equivalent complex with 1.0f as the imaginary part
            Eigen::Tensor4dXcf vjtmp_complex = vjtmp.unaryExpr([](float x) {return std::complex<float>(x, 1.0f);});

            // create a Wiener gain for this source
            Eigen::Tensor4dXcf filter = vjtmp_complex * R[j] * Cxx;

            // apply it to the mixture
            y = _apply_filter(y, x, filter, nb_sources, nb_channels);
        }
    }

    return y;
}
