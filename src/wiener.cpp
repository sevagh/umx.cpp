#include "wiener.hpp"
#include <iostream>

// torch.sqrt(_norm(mix_stft)).max()
static float max_sqrt_norm(const umxcpp::StereoSpectrogramC &spectrogram) {
    float max_sqrt_norm = 0.0;
    for (int i = 0; i < spectrogram.left.size(); i++) {
        for (int j = 0; j < spectrogram.left[0].size(); j++) {
            auto sqrt_norm_left = std::sqrt(std::norm(spectrogram.left[i][j]));
            auto sqrt_norm_right = std::sqrt(std::norm(spectrogram.right[i][j]));

            max_sqrt_norm = std::max(max_sqrt_norm, std::max(sqrt_norm_left, sqrt_norm_right));
        }
    }
    return max_sqrt_norm;
}

// forward declarations
static std::array<umxcpp::StereoSpectrogramC, 4> expectation_maximization(
    std::array<umxcpp::StereoSpectrogramC, 4> &y,
    umxcpp::StereoSpectrogramC &mix_stft
);
static Eigen::MatrixXcf _invert(const Eigen::MatrixXcf& M);
static std::vector<Eigen::MatrixXcf> _covariance(const std::vector<Eigen::MatrixXcf>& y_j);

static const float EPS = 1e-10;

std::array<umxcpp::StereoSpectrogramC, 4> umxcpp::wiener_filter(
    const umxcpp::StereoSpectrogramC &spectrogram,
    const umxcpp::StereoSpectrogramC (&targets)[4]
) {
    std::array<umxcpp::StereoSpectrogramC, 4> y;
    for (int i = 0; i < 4; i++) {
        y[i] = targets[i];
    }

    umxcpp::StereoSpectrogramC mix_stft = spectrogram;

    const float scale_factor = 10.0;

    std::cout << "A" << std::endl;

    auto max_abs = std::max(
       1.0f,
       max_sqrt_norm(spectrogram) / scale_factor);

    std::cout << "B" << std::endl;

    // mix_stft = mix_stft / max_abs
    // y = y / max_abs
    for (int i = 0; i < y[0].left.size(); ++i) {
        for (int j = 0; j < y[0].left[0].size(); ++j) {
            mix_stft.left[i][j] /= max_abs;
            mix_stft.right[i][j] /= max_abs;

            for (int k = 0; k < 4; ++k) {
                y[k].left[i][j] /= max_abs;
                y[k].right[i][j] /= max_abs;
            }
        }
    }
    std::cout << "C" << std::endl;

    auto y_em = expectation_maximization(y, mix_stft);

    std::cout << "D" << std::endl;

    // scale estimates up again
    // y = y * max_abs
    for (int i = 0; i < y[0].left.size(); ++i) {
        for (int j = 0; j < y[0].left[0].size(); ++j) {
            for (int k = 0; k < 4; ++k) {
                y_em[k].left[i][j] *= max_abs;
                y_em[k].right[i][j] *= max_abs;
            }
        }
    }

    std::cout << "E" << std::endl;

    return y_em;
}

static std::array<umxcpp::StereoSpectrogramC, 4> expectation_maximization(
    std::array<umxcpp::StereoSpectrogramC, 4> &y,
    umxcpp::StereoSpectrogramC &mix_stft
) {
    const int nb_sources = 4;
    const int nb_channels = 2;
    const int batch_size = 200;
    int nb_bins = mix_stft.left[0].size();
    int nb_frames = mix_stft.left.size();

        // Regularization term: in this case, it's a diagonal matrix with size nb_channels
    Eigen::MatrixXcf regularization = Eigen::MatrixXcf::Identity(nb_channels, nb_channels);

    // Initialize R
    std::vector<Eigen::MatrixXcf> R(nb_sources, Eigen::MatrixXcf::Zero(nb_bins, nb_channels));

    // Initialize weights
    Eigen::VectorXcf weight = Eigen::VectorXcf::Zero(nb_bins);

    // Compute power spectrogram |y|^2 and average over channels
    std::vector<Eigen::MatrixXcf> v(nb_sources, Eigen::MatrixXcf::Zero(nb_frames, nb_bins));

    for (int t = 0; t < nb_frames; t++) {
        for (int j = 0; j < nb_sources; j++) {
            // Compute |y|^2 for each source and add them up
            // assuming y[t][j] returns the complex spectrogram for source j at time frame t
            Eigen::VectorXcf y_t_j_l(nb_bins);
            Eigen::VectorXcf y_t_j_r(nb_bins);

            // copy from std::vector<std::complex<float>> into vectors above
            for (int i = 0; i < nb_bins; ++i) {
                y_t_j_l[i] = y[j].left[t][i];
                y_t_j_r[i] = y[j].right[t][i];
            }

            Eigen::VectorXcf y_t_j_l_sq = y_t_j_l.cwiseProduct(y_t_j_l.conjugate());
            Eigen::VectorXcf y_t_j_r_sq = y_t_j_r.cwiseProduct(y_t_j_r.conjugate());

            v[j](t) = (y_t_j_l_sq + y_t_j_r_sq).mean();
        }
    }

    std::cout << "EM F" << std::endl;

    // Compute covariance matrices
    for (int j = 0; j < nb_sources; j++) {
        std::cout << "EM F1 " << j << std::endl;

        weight.setZero();

        std::cout << "EM F2 " << j << std::endl;

        for (int pos = 0; pos < nb_frames; pos += batch_size) {
            std::cout << "why this only once?" << std::endl;
            std::cout << "EM F2 1 " << j << " " << pos << std::endl;

            // Take a batch
            int t = std::min(nb_frames, pos + batch_size);

            std::cout << "EM F2 2 " << j << " " << pos << std::endl;

            // Add up covariance matrices
            // y[t,...,j] ->
            // y[j].left[t], y[j].right[t]
            std::cout << "EM F2 3 " << j << " " << pos << std::endl;
            std::vector<Eigen::MatrixXcf> y_slices(t);
            for (int i = 0; i < t; ++i) {
                y_slices[i] = Eigen::MatrixXcf(nb_bins, nb_channels);

                for (int k = 0; k < nb_bins; ++k) {
                    y_slices[i](k, 0) = y[j].left[i][k];
                    y_slices[i](k, 1) = y[j].right[i][k];
                }
            }

            std::cout << "EM F2 4 " << j << " " << pos << std::endl;
            //R[j] += _covariance(y_slices).sum();
            auto cov = _covariance(y_slices);
            std::cout << "R size: " << R[j].rows() << " " << R[j].cols() << std::endl;
            std::cout << "cov size: " << cov.size() << " " << cov[0].rows() << " " << cov[0].cols() << std::endl;
            for (int i = 0; i < cov.size()/nb_frames; ++i) {
                // sum of cov over the first dimension
                R[j] += cov[i];
            }
            std::cout << "EM F2 5 " << j << " " << pos << std::endl;

            // Add up weights
            for (int i = 0; i < v[j].rows(); ++i) {
                weight(i) += v[j](i);
            }
            std::cout << "EM F2 6 " << j << " " << pos << std::endl;
            std::cout << "what the fuck" << std::endl;
        }
        std::cout << "something is weird" << std::endl;

        std::cout << "EM F2 7 " << j << std::endl;

        // Normalize covariance matrices
        // broadcast weight to fit R[j]
        // R[j] /= weight;
        for (int i = 0; i < R[j].rows(); ++i) {
            for (int k = 0; k < R[j].cols(); ++k) {
                R[j](i, k) /= weight[i];
            }
        }
        std::cout << "EM F2 8 " << j << std::endl;
    }

    std::cout << "EM G" << std::endl;

    // Update estimates
    for (int pos = 0; pos < nb_frames; pos += batch_size) {
        // Take a batch
        int t = std::min(nb_frames, pos + batch_size);

        // Reset y
        for (int j = 0; j < nb_sources; j++) {
            //y[pos][j].left.setZero();
            //y[pos][j].right.setZero();
            std::fill(y[j].left[pos].begin(), y[j].left[pos].end(), 0);
            std::fill(y[j].right[pos].begin(), y[j].right[pos].end(), 0);
        }

        // Compute mixture covariance matrix
        Eigen::MatrixXcf Cxx = regularization;

        for (int j = 0; j < nb_sources; j++) {
            Cxx += v[j].block(pos, 0, t - pos, nb_bins).asDiagonal() * R[j];
        }

        // Invert the mixture covariance matrix
        Eigen::MatrixXcf inv_Cxx = _invert(Cxx);

        // Update estimates for each source
        for (int j = 0; j < nb_sources; j++) {
            // Compute Wiener gain
            Eigen::MatrixXcf gain = inv_Cxx * R[j] * v[j].block(pos, 0, t - pos, nb_bins).asDiagonal();

            // create two Eigen::MatrixXcf to store mix_stft left and right
            Eigen::MatrixXcf mix_stft_l(t - pos, nb_bins);
            Eigen::MatrixXcf mix_stft_r(t - pos, nb_bins);

            // copy from std::vector<std::complex<float>> into vectors above
            for (int i = 0; i < nb_bins; ++i) {
                for (int k = 0; k < t - pos; ++k) {
                    mix_stft_l(k, i) = mix_stft.left[k + pos][i];
                    mix_stft_r(k, i) = mix_stft.right[k + pos][i];
                }
            }

            // Apply gain to mixture
            auto gain_l = gain * mix_stft_l;
            auto gain_r = gain * mix_stft_r;

            // now copy into std::vector<std::complex<float>> y[j].left[pos] and y[j].right[pos]
            for (int i = 0; i < nb_bins; ++i) {
                for (int k = 0; k < t - pos; ++k) {
                    y[j].left[k + pos][i] = gain_l(k, i);
                    y[j].right[k + pos][i] = gain_r(k, i);
                }
            }
        }
    }

    return y;
}

static std::complex<float> _inv(const std::complex<float>& z) {
    float norm = std::norm(z);
    return std::complex<float>(z.real() / norm, -z.imag() / norm);
}


// Function to invert a complex matrix
static Eigen::MatrixXcf _invert(const Eigen::MatrixXcf& M) {
    int nb_channels = M.rows();

    Eigen::MatrixXcf invM;
    if(nb_channels == 1){
        // scalar case
        invM(0,0) = _inv(M(0,0));
    }
    else if(nb_channels == 2){
        // 2x2 case: analytical expression

        // Compute the determinant
        std::complex<float> det = M(0,0) * M(1,1) - M(0,1) * M(1,0);

        // Invert the determinant
        std::complex<float> invDet = _inv(det);

        // Fill out the matrix with the inverse
        invM(0,0) = invDet * M(1,1);
        invM(1,0) = -invDet * M(1,0);
        invM(0,1) = -invDet * M(0,1);
        invM(1,1) = invDet * M(0,0);
    }
    else{
        // Throw exception
        throw std::invalid_argument("Only 1 or 2 channels are supported.");
    }
    return invM;
}

static std::vector<Eigen::MatrixXcf> _covariance(const std::vector<Eigen::MatrixXcf>& y_j) {
    int nb_frames = y_j.size();
    int nb_bins = y_j[0].rows();
    int nb_channels = y_j[0].cols();

    std::vector<Eigen::MatrixXcf> Cj(nb_frames * nb_bins, Eigen::MatrixXcf::Zero(nb_channels, nb_channels));

    for(int i = 0; i < nb_frames; ++i) {
        for(int j = 0; j < nb_bins; ++j) {
            for(int ch1 = 0; ch1 < nb_channels; ++ch1) {
                for(int ch2 = 0; ch2 < nb_channels; ++ch2) {
                    Cj[i * nb_bins + j](ch1, ch2) += y_j[i](j, ch1) * std::conj(y_j[i](j, ch2));
                }
            }
        }
    }
    return Cj;
}
