#ifndef LSTM_HPP
#define LSTM_HPP

#include "model.hpp"
#include <Eigen/Dense>

namespace umxcpp
{

Eigen::MatrixXf umx_lstm_forward(const struct umx_model &model, int target,
                                 const Eigen::MatrixXf &input, int hidden_size);

}; // namespace umxcpp

#endif // LSTM_HPP
