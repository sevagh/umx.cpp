#include "wiener.hpp"
#include "dsp.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorChipping.h>
#include <vector>
#include <array>
#include <cassert>
#include <tuple>

static inline float mulAdd(float a, float b, float c) {
    return a * b + c;
}

// Function to compute the absolute maximum value from a complex 2D vector
static float find_max_abs(const Eigen::Tensor3dXcf &data, float scale_factor) {
    float max_val_im = -1.0f;
    for (int i = 0; i < data.dimension(0); ++i) {
        for (int j = 0; j < data.dimension(1); ++j) {
            for (int k = 0; k < data.dimension(2); ++k) {
                max_val_im = std::max(max_val_im, std::sqrt(std::norm(data(i, j, k))));
            }
        }
    }
    return std::max(1.0f, max_val_im/scale_factor);
}

static std::complex<float> inv(std::complex<float> z) {
    const float threshold = 1e-6;
    float ez = std::norm(z); // Compute the norm (magnitude squared)

    if (std::abs(ez) < threshold) {
        return std::complex<float>(1.0 / threshold, 0);  // or whatever large value you find appropriate
    }
    return std::conj(z) / ez;
}

static void invertMatrix(umxcpp::Tensor3D& M) {
    std::cout << "invert!" << std::endl;
    for (std::size_t ch1 = 0; ch1 < M.data.size(); ++ch1) {
        std::cout << "iter: " << ch1 << std::endl;

        std::cout << "initialize floats" << std::endl;

        std::complex<float> a(M.data[ch1][0][0], M.data[ch1][0][1]);
        std::complex<float> b(M.data[ch1][0][1], M.data[ch1][0][1]);
        std::complex<float> c(M.data[ch1][1][0], M.data[ch1][1][0]);
        std::complex<float> d(M.data[ch1][1][1], M.data[ch1][1][1]);

        std::cout << "compute det" << std::endl;

        // Compute the determinant
        std::complex<float> det = a * d - b * c;
        std::complex<float> invDet = inv(det);

        std::cout << "invert 2x2" << std::endl;

        // Invert the 2x2 matrix
        std::complex<float> tmp00 = invDet * d;
        std::complex<float> tmp01 = -invDet * b;
        std::complex<float> tmp10 = -invDet * c;
        std::complex<float> tmp11 = invDet * a;

        std::cout << "update floats" << std::endl;

        // Update the original tensor
        M.data[ch1][0][0] = tmp00.real();
        M.data[ch1][0][1] = tmp00.imag();
        M.data[ch1][1][0] = tmp10.real();
        M.data[ch1][1][1] = tmp10.imag();
        M.data[ch1][0][1] = tmp01.real();
        M.data[ch1][1][0] = tmp01.imag();  // Fixed this line
        M.data[ch1][1][1] = tmp11.real();
        M.data[ch1][1][1] = tmp11.imag();

        std::cout << "yay!" << std::endl;
    }
    std::cout << "Successful invert!" << std::endl;
}

// Compute the empirical covariance for a source.
// forward decl
static umxcpp::Tensor5D calculateCovariance(
    const Eigen::Tensor3dXcf &y_j,
    const int pos,
    const int t_end
);

static umxcpp::Tensor4D sumAlongFirstDimension(const umxcpp::Tensor5D& tensor5d) {
    int nb_frames = tensor5d.data.size();
    int nb_bins = tensor5d.data[0].size();
    int nb_channels1 = tensor5d.data[0][0].size();
    int nb_channels2 = tensor5d.data[0][0][0].size();
    int nb_reim = tensor5d.data[0][0][0][0].size();

    // Initialize a 4D tensor filled with zeros
    umxcpp::Tensor4D result(nb_bins, nb_channels1, nb_channels2, nb_reim);

    for (int frame = 0; frame < nb_frames; ++frame) {
        for (int bin = 0; bin < nb_bins; ++bin) {
            for (int ch1 = 0; ch1 < nb_channels1; ++ch1) {
                for (int ch2 = 0; ch2 < nb_channels2; ++ch2) {
                    for (int reim = 0; reim < nb_reim; ++reim) {
                        result.data[bin][ch1][ch2][reim] += tensor5d.data[frame][bin][ch1][ch2][reim];
                    }
                }
            }
        }
    }

    return result;
}

// Wiener filter function
std::array<Eigen::Tensor3dXcf, 4>
umxcpp::wiener_filter(Eigen::Tensor3dXcf &mix_stft,
              const std::array<Eigen::Tensor3dXf, 4> &targets_mag_spectrograms)
{
    // first just do naive mix-phase
    std::array<Eigen::Tensor3dXcf, 4> y;

    Eigen::Tensor3dXf mix_phase = mix_stft.unaryExpr(
        [](const std::complex<float> &c) { return std::arg(c); });

    for (int target = 0; target < 4; ++target) {
        y[target] = umxcpp::polar_to_complex(targets_mag_spectrograms[target], mix_phase);
    }

    // we need to refine the estimates. Scales down the estimates for
    // numerical stability
    float max_abs = find_max_abs(mix_stft, WIENER_SCALE_FACTOR);

    // Dividing mix_stft by max_abs
    for (int i = 0; i < mix_stft.dimension(1); ++i) {
        for (int j = 0; j < mix_stft.dimension(2); ++j) {
            mix_stft(0, i, j) = std::complex<float>{
                    mix_stft(0, i, j).real()/max_abs,
                    mix_stft(0, i, j).imag()/max_abs};
            mix_stft(1, i, j) = std::complex<float>{
                    mix_stft(1, i, j).real()/max_abs,
                    mix_stft(1, i, j).imag()/max_abs};
        }
    }

    // Dividing y by max_abs
    for (int source = 0; source < 4; ++source) {
        for (int i = 0; i < mix_stft.dimension(1); ++i) {
            for (int j = 0; j < mix_stft.dimension(2); ++j) {
                y[source](0, i, j) = std::complex<float>{
                        y[source](0, i, j).real()/max_abs,
                        y[source](0, i, j).imag()/max_abs};
                y[source](1, i, j) = std::complex<float>{
                        y[source](1, i, j).real()/max_abs,
                        y[source](1, i, j).imag()/max_abs};
            }
        }
    }

    // call expectation maximization
    // y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]

    const int nb_channels = 2;
    const int nb_frames = mix_stft.dimension(1);
    const int nb_bins = mix_stft.dimension(2);
    const int nb_sources = 4;
    const float eps = WIENER_EPS;

    // Create and initialize the 5D tensor
    umxcpp::Tensor3D regularization(nb_channels, nb_channels, 2); // The 3D tensor
    // Fill the diagonal with sqrt(eps) for all 3D slices in dimensions 0 and 1
    regularization.fill_diagonal(std::sqrt(eps));

    std::vector<Tensor4D> R; // A vector to hold each source's covariance matrix
    for (int j = 0; j < nb_sources; ++j) {
        R.emplace_back(Tensor4D(nb_bins, nb_channels, nb_channels, 2));
    }

    Tensor1D weight(nb_bins);  // A 1D tensor (vector) of zeros
    Tensor3D v(nb_frames, nb_bins, nb_sources);  // A 3D tensor of zeros

    for (int it = 0; it < WIENER_ITERATIONS; ++it) {
        std::cout << "wiener iter: " << it << std::endl;

        // update the PSD as the average spectrogram over channels
        // PSD container is v
        for (int frame = 0; frame < nb_frames; ++frame) {
            for (int bin = 0; bin < nb_bins; ++bin) {
                for (int source = 0; source < nb_sources; ++source) {
                    float sumSquare = 0.0f;
                    for (int channel = 0; channel < nb_channels; ++channel) {
                        float realPart = 0.0f;
                        float imagPart = 0.0f;

                        realPart += y[source](channel, frame, bin).real();
                        imagPart += y[source](channel, frame, bin).imag();

                        sumSquare += (realPart * realPart) + (imagPart * imagPart);
                    }
                    // Divide by the number of channels to get the average
                    v.data[frame][bin][source] = sumSquare / nb_channels;
                }
            }
        }

        std::cout << "begin update spatial covariance matrices R" << std::endl;
        for (int source = 0; source < nb_sources; ++source) {
            R[source].setZero();  // Assume Tensor4d has a method to set all its elements to zero
            weight.fill(WIENER_EPS); // Initialize with small epsilon (assume Tensor1d has a fill method)

            int pos = 0;
            int batchSize = WIENER_EM_BATCH_SIZE > 0 ? WIENER_EM_BATCH_SIZE : nb_frames;

            while (pos < nb_frames) {
                int t_end = std::min(nb_frames, pos + batchSize);

                umxcpp::Tensor5D tempR = calculateCovariance(y[source], pos, t_end);

                // Sum the calculated covariance into R[j]
                // Sum along the first (time/frame) dimension to get a 4D tensor
                umxcpp::Tensor4D tempR4D = sumAlongFirstDimension(tempR);

                // Add to existing R[j]; (R[j], tempR4D have the same dimensions)
                for (std::size_t bin = 0; bin < R[source].data.size(); ++bin) {
                    for (std::size_t ch1 = 0; ch1 < R[source].data[0].size(); ++ch1) {
                        for (std::size_t ch2 = 0; ch2 < R[source].data[0][0].size(); ++ch2) {
                            for (std::size_t reim = 0; reim < R[source].data[0][0][0].size(); ++reim) {
                                R[source].data[bin][ch1][ch2][reim] += tempR4D.data[bin][ch1][ch2][reim];
                            }
                        }
                    }
                }

                // Update the weight summed v values across the frames for this batch
                for (int t = pos; t < t_end; ++t) {
                    for (int bin = 0; bin < nb_bins; ++bin) {
                        weight.data[bin] += v.data[t][bin][source];
                    }
                }

                pos = t_end;
            }

            // Normalize R[j] by weight
            for (int bin = 0; bin < nb_bins; ++bin) {
                for (int ch1 = 0; ch1 < nb_channels; ++ch1) {
                    for (int ch2 = 0; ch2 < nb_channels; ++ch2) {
                        for (int k = 0; k < 2; ++k) {
                            R[source].data[bin][ch1][ch2][k] /= weight.data[bin];
                        }
                    }
                }
            }

            // Reset the weight for the next iteration
            weight.fill(0.0f);
        }
        std::cout << "end update spatial covariance matrices R" << std::endl;

        // so far so good!
        std::cout << "WIENER GOOD, MATCH SO FAR! regularization, R, v, covariance!" << std::endl;

        int pos = 0;
        int batchSize = WIENER_EM_BATCH_SIZE > 0 ? WIENER_EM_BATCH_SIZE : nb_frames;
        std::cout << "while pos < nb_frames loop 2" << std::endl;
        while (pos < nb_frames) {
            int t_end = std::min(nb_frames, pos + batchSize);

            // Reset y values to zero for this batch
            // Assuming you have a way to set all elements of y between frames pos and t_end to 0.0
            for (int source = 0; source < 4; ++source) {
                for (int i = pos; i < t_end; ++i) {
                     for (int j = 0; j < nb_bins; ++j) {
                         y[source](0, i, j) = std::complex<float>{0.0f, 0.0f};
                         y[source](1, i, j) = std::complex<float>{0.0f, 0.0f};
                     }
                }
            }

            // Compute mix covariance matrix Cxx
            //Tensor5D Cxx = regularization; // Assuming copy constructor or assignment operator performs deep copy
            Tensor3D Cxx = regularization;

            for (int source = 0; source < nb_sources; ++source) {
                for (int t = pos; t < t_end; ++t) {
                    for (int bin = 0; bin < nb_bins; ++bin) {
                        float multiplier = v.data[t][bin][source];
                        // Element-wise addition and multiplication to update Cxx
                        for (int ch1 = 0; ch1 < nb_channels; ++ch1) {
                            for (int ch2 = 0; ch2 < nb_channels; ++ch2) {
                                for (int re_im = 0; re_im < 2; ++re_im) {
                                    Cxx.data[ch1][ch2][re_im] += multiplier * R[source].data[bin][ch1][ch2][re_im];
                                }
                            }
                        }
                    }
                }
            }

            float Cxx_min = 1000000000000.0f;
            float Cxx_max = -1000000000000.0f;

            for (int ch1 = 0; ch1 < nb_channels; ++ch1) {
                for (int ch2 = 0; ch2 < nb_channels; ++ch2) {
                    for (int re_im = 0; re_im < 2; ++re_im) {
                        Cxx_min = std::min(Cxx_min, Cxx.data[ch1][ch2][re_im]);
                        Cxx_max = std::max(Cxx_max, Cxx.data[ch1][ch2][re_im]);
                    }
                }
            }

            std::cout << "mix covariance matrix Cxx: " << Cxx_min << ", " << Cxx_max << std::endl;

            std::cin.ignore();

            std::cout << "where the fuck are we 2" << std::endl;

            // Invert Cxx
            invertMatrix(Cxx);  // Assuming invertMatrix performs element-wise inversion
            Tensor3D inv_Cxx = Cxx;  // Assuming copy constructor or assignment operator performs deep copy

            std::cout << "where the fuck are we 3" << std::endl;

            // Separate the sources
            for (int source = 0; source < nb_sources; ++source) {
                std::cout << "source: " << source << std::endl;

                // Initialize with zeros
                // create gain with broadcast size of inv_Cxx
                Tensor5D gain(nb_frames, nb_bins, nb_channels, nb_channels, 2);
                gain.setZero();

                std::cout << "loop 1" << std::endl;
                for (int frame = 0; frame < nb_frames; ++frame) {
                    for (int bin = 0; bin < nb_bins; ++bin) {
                        for (int ch1 = 0; ch1 < nb_channels; ++ch1) {
                            for (int ch2 = 0; ch2 < nb_channels; ++ch2) {
                                for (int re_im = 0; re_im < 2; ++re_im) { // Assuming last dimension has size 2 (real/imaginary)
                                    for (int ch3 = 0; ch3 < nb_channels; ++ch3) {
                                        gain.data[frame][bin][ch1][ch2][re_im] = mulAdd(
                                            R[source].data[bin][ch1][ch3][re_im],
                                            inv_Cxx.data[ch3][ch2][re_im],   // implicit broadcasting
                                            gain.data[frame][bin][ch1][ch2][re_im] // explicit broadcasting to have independent gains
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                std::cout << "loop 2" << std::endl;
                // Element-wise multiplication with v
                for (int t = pos; t < t_end; ++t) {
                    std::cout << "t: " << t << std::endl;
                    for (int frame = 0; frame < nb_frames; ++frame) {
                        for (int bin = 0; bin < nb_bins; ++bin) {
                            for (int ch1 = 0; ch1 < nb_channels; ++ch1) {
                                for (int ch2 = 0; ch2 < nb_channels; ++ch2) {
                                    for (int re_im = 0; re_im < 2; ++re_im) { // Assuming last dimension has size 2 (real/imaginary)
                                        float multiplier = v.data[frame][bin][source];
                                        gain.data[frame][bin][ch1][ch2][re_im] *= multiplier;
                                    }
                                }
                            }
                        }
                    }
                }

                std::cout << "loop 3" << std::endl;
                for (int frame = pos; frame < t_end; ++frame) {
                    for (int bin = 0; bin < nb_bins; ++bin) {
                        for (int i = 0; i < nb_channels; ++i) {
                            for (int source = 0; source < nb_sources; ++source) {
                                // Assume x.data and y.data have dimensions [frame][bin][channel][re_im]
                                // and gain.data has dimensions [frame][bin][ch1][ch2][re_im]
                                float left_real = mulAdd(
                                    gain.data[frame][bin][i][i][0],  // assuming we use the same channel i for gain
                                    mix_stft(0, frame, bin).real(),
                                    y[source](0, frame, bin).real()
                                );
                                float right_real = mulAdd(
                                    gain.data[frame][bin][i][i][0],  // assuming we use the same channel i for gain
                                    mix_stft(1, frame, bin).real(),
                                    y[source](1, frame, bin).real()
                                );
                                float left_im = mulAdd(
                                    gain.data[frame][bin][i][i][1],  // assuming we use the same channel i for gain
                                    mix_stft(0, frame, bin).imag(),
                                    y[source](0, frame, bin).imag()
                                );
                                float right_im = mulAdd(
                                    gain.data[frame][bin][i][i][1],  // assuming we use the same channel i for gain
                                    mix_stft(1, frame, bin).imag(),
                                    y[source](1, frame, bin).imag()
                                );

                                y[source](0, frame, bin) = std::complex<float>(left_real, left_im);
                                y[source](1, frame, bin) = std::complex<float>(right_real, right_im);
                            }
                        }
                    }
                }
            }

            pos = t_end;
        }
    }

    // scale y by max_abs again
    for (int source = 0; source < 4; ++source) {
        for (int i = 0; i < mix_stft.dimension(1); ++i) {
            for (int j = 0; j < mix_stft.dimension(2); ++j) {
                y[source](0, i, j) = std::complex<float>{
                    y[source](0, i, j).real()*max_abs,
                    y[source](0, i, j).imag()*max_abs};
                y[source](1, i, j) = std::complex<float>{
                    y[source](1, i, j).real()*max_abs,
                    y[source](1, i, j).imag()*max_abs};
            }
        }

        if (source == 0) {
            std::cout << "y_debug: " << y[0](0, 14, 587).real() << ", " << y[0](0, 14, 587).imag() << std::endl;
            std::cout << "y_debug: " << y[0](0, 23, 1023).real() << ", " << y[0](0, 23, 1023).imag() << std::endl;
        }
    }

    return y;
}

// Compute the empirical covariance for a source.
/*
 *   y_j shape: 2, nb_frames_total, 2049
 *   pos-t_end = nb_frames (i.e. a chunk of y_j)
 *
 *   returns Cj:
 *       shape: nb_frames, nb_bins, nb_channels, nb_channels, realim
 */
static umxcpp::Tensor5D calculateCovariance(
    const Eigen::Tensor3dXcf &y_j,
    const int pos,
    const int t_end
) {
    //int nb_frames = y_j.dimension(1);
    int nb_frames = t_end-pos;
    int nb_bins = y_j.dimension(2);
    int nb_channels = 2;

    // Initialize Cj tensor with zeros
    umxcpp::Tensor5D Cj(nb_frames, nb_bins, nb_channels, nb_channels, 2);
    Cj.setZero();

    for (int frame = 0; frame < nb_frames; ++frame) {
        for (int bin = 0; bin < nb_bins; ++bin) {
            for (int ch1 = 0; ch1 < nb_channels; ++ch1) {
                for (int ch2 = 0; ch2 < nb_channels; ++ch2) {
                    // assign real
                    std::complex<float> a = y_j(ch1, frame+pos, bin);
                    std::complex<float> b = std::conj(y_j(ch2, frame+pos, bin));

                    float a_real = a.real();
                    float a_imag = a.imag();

                    float b_real = b.real();
                    float b_imag = b.imag();

                    // Update the tensor
                    // _mul_add y_j, conj(y_j) -> y_j = a, conj = b
                    Cj.data[frame][bin][ch1][ch2][0] += (a_real*b_real - a_imag*b_imag);
                    Cj.data[frame][bin][ch1][ch2][1] += (a_real*b_imag + a_imag*b_real);
                }
            }
        }
    }

    return Cj;
}
