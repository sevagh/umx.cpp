#include "wiener.hpp"

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

std::array<umxcpp::StereoSpectrogramC, 4> umxcpp::wiener_filter(
    const umxcpp::StereoSpectrogramC &spectrogram,
    const umxcpp::StereoSpectrogramC (&targets)[4]
) {
    std::array<umxcpp::StereoSpectrogramC, 4> y;
    for (int i = 0; i < 4; i++) {
        y[i] = targets[i];
    }

    umxcpp::StereoSpectrogramC mix_stft = spectrogram;

    const float eps = 1e-10;
    const float scale_factor = 10.0;
    // residual = false, softmask = false, niter = 1

    /*
    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = torch.max(
        torch.as_tensor(1.0, dtype=mix_stft.dtype, device=mix_stft.device),
         / scale_factor,
    )
    */
    auto max_abs = std::max(
       1.0f,
       max_sqrt_norm(spectrogram) / scale_factor);

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

    /*
    # call expectation maximization
    y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]
    */

    // scale estimates up again
    // y = y * max_abs
    for (int i = 0; i < y[0].left.size(); ++i) {
        for (int j = 0; j < y[0].left[0].size(); ++j) {
            for (int k = 0; k < 4; ++k) {
                y[k].left[i][j] *= max_abs;
                y[k].right[i][j] *= max_abs;
            }
        }
    }

    return y;
}
