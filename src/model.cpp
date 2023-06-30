#include "model.hpp"
#include "dsp.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <unsupported/Eigen/FFT>
#include <vector>

// forward declaration
static size_t load_single_matrix(FILE *f, std::string &name,
                                 Eigen::MatrixXf &matrix, int ne[2],
                                 int32_t nelements);

// from scripts/convert-pth-to-ggml.py
bool umxcpp::load_umx_model(const std::string &model_dir,
                            struct umx_model *model)
{
    fprintf(stderr, "%s: loading model\n", __func__);

    // equivalent of os.listdir(model_dir) in C++
    std::vector<std::string> model_files;
    for (const auto &entry : std::filesystem::directory_iterator(model_dir))
    {
        std::cout << "Discovered model file " << entry.path() << " in model dir"
                  << model_dir << std::endl;
        model_files.push_back(entry.path());
    }

    // sort the vector of file paths
    // to ensure that the order of the targets is consistent
    std::sort(model_files.begin(), model_files.end());

    // compute t_start_us using C++ std::chrono
    const auto t_start_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    uint32_t hidden_size = 0;

    // verify magic and hidden size
    {
        uint32_t magic;
        uint32_t hidden_size_tmp;

        // equivalent of with open(...) as f on each model_file
        for (const auto &model_file : model_files)
        {
            std::cout << "Checking the magic of model_file " << model_file
                      << std::endl;

            FILE *f = fopen(model_file.c_str(), "rb");
            if (!f)
            {
                fprintf(stderr, "%s: failed to open %s\n", __func__,
                        model_file.c_str());
                return false;
            }

            // read the size of uint32_t bytes from f into magic
            fread(&magic, sizeof(uint32_t), 1, f);
            if (magic != 0x756d7867)
            {
                fprintf(stderr, "%s: invalid model data (bad magic)\n",
                        __func__);
                return false;
            }

            // read the size of uint32_t bytes from f into hidden_size_tmp
            fread(&hidden_size_tmp, sizeof(uint32_t), 1, f);
            if (hidden_size == 0)
            {
                hidden_size = hidden_size_tmp;
            }
            else if (hidden_size != hidden_size_tmp)
            {
                fprintf(stderr,
                        "%s: invalid model data (mismatched hidden size %u vs. "
                        "%u)\n",
                        __func__, hidden_size_tmp, hidden_size);
                return false;
            }

            fclose(f);
        }
    }

    std::cout << "Loaded umx model with hidden size " << hidden_size
              << std::endl;

    model->hidden_size = hidden_size;

    // loaded tensor shapes
    //    Processing variable:  fc1.weight  with shape:  (HIDDEN, 2974)
    //    Processing variable:  bn1.{weight, bias}  with shape:  (HIDDEN,)
    //    Processing variable:  lstm.weight_ih_l{0,1,2}  with shape:  (2*HIDDEN,
    //    HIDDEN) Processing variable:  lstm.weight_hh_l{0,1,2}  with shape:
    //    (2*HIDDEN, HIDDEN/2) Processing variable:  lstm.bias_ih_l{0,1,2}  with
    //    shape:  (2*HIDDEN,) Processing variable:  lstm.bias_hh_l{0,1,2}  with
    //    shape:  (2*HIDDEN,) Processing variable:
    //    lstm.weight_ih_l{0,1,2}_reverse  with shape:  (2*HIDDEN, HIDDEN)
    //    Processing variable:  lstm.weight_hh_l{0,1,2}_reverse  with shape:
    //    (2*HIDDEN, HIDDEN/2) Processing variable:
    //    lstm.bias_ih_l{0,1,2}_reverse  with shape:  (2*HIDDEN,) Processing
    //    variable:  lstm.bias_hh_l{0,1,2}_reverse  with shape:  (2*HIDDEN,)
    //    Processing variable:  fc2.weight  with shape:  (HIDDEN, 2*HIDDEN)
    //    Processing variable:  bn2.weight  with shape:  (HIDDEN,)
    //    Processing variable:  bn2.bias  with shape:  (HIDDEN,)
    //    Processing variable:  fc3.weight  with shape:  (4098, HIDDEN)
    //    Processing variable:  bn3.weight  with shape:  (4098,)
    //    Processing variable:  bn3.bias  with shape:  (4098,)

    auto lstm_size_1 = 2 * hidden_size;
    auto lstm_size_2 = hidden_size / 2;

    // prepare memory for the weights
    {
        for (int target = 0; target < 4; ++target)
        {
            model->input_mean[target] = Eigen::MatrixXf(2 * 1487, 1);
            model->input_scale[target] = Eigen::MatrixXf(2 * 1487, 1);
            model->output_mean[target] = Eigen::MatrixXf(2 * 2049, 1);
            model->output_scale[target] = Eigen::MatrixXf(2 * 2049, 1);

            // fc1, fc2, fc3
            model->fc1_w[target] = Eigen::MatrixXf(2974, hidden_size);
            model->fc2_w[target] = Eigen::MatrixXf(lstm_size_1, hidden_size);
            model->fc3_w[target] = Eigen::MatrixXf(hidden_size, 4098);

            // bn1, bn2, bn3
            model->bn1_w[target] = Eigen::MatrixXf(hidden_size, 1);
            model->bn1_b[target] = Eigen::MatrixXf(hidden_size, 1);
            model->bn1_rm[target] = Eigen::MatrixXf(hidden_size, 1);
            model->bn1_rv[target] = Eigen::MatrixXf(hidden_size, 1);

            model->bn2_w[target] = Eigen::MatrixXf(hidden_size, 1);
            model->bn2_b[target] = Eigen::MatrixXf(hidden_size, 1);
            model->bn2_rm[target] = Eigen::MatrixXf(hidden_size, 1);
            model->bn2_rv[target] = Eigen::MatrixXf(hidden_size, 1);

            model->bn3_w[target] = Eigen::MatrixXf(4098, 1);
            model->bn3_b[target] = Eigen::MatrixXf(4098, 1);
            model->bn3_rm[target] = Eigen::MatrixXf(4098, 1);
            model->bn3_rv[target] = Eigen::MatrixXf(4098, 1);

            // 3 layers of lstm
            for (int lstm_layer = 0; lstm_layer < 3; ++lstm_layer)
            {
                for (int direction = 0; direction < 2; ++direction)
                {
                    model->lstm_ih_w[target][lstm_layer][direction] =
                        Eigen::MatrixXf(hidden_size, lstm_size_1);
                    model->lstm_hh_w[target][lstm_layer][direction] =
                        Eigen::MatrixXf(lstm_size_2, lstm_size_1);
                    model->lstm_ih_b[target][lstm_layer][direction] =
                        Eigen::MatrixXf(lstm_size_1, 1);
                    model->lstm_hh_b[target][lstm_layer][direction] =
                        Eigen::MatrixXf(lstm_size_1, 1);
                }
            }
        }
    }

    size_t total_size = 0;
    uint32_t n_loaded = 0;

    // load weights
    {
        // equivalent of with open(...) as f on each model_file
        int target_counter = 0;
        for (const auto &model_file : model_files)
        {
            std::cout << "Loading weights from model_file " << model_file
                      << " into target " << target_counter << std::endl;

            FILE *f = fopen(model_file.c_str(), "rb");
            if (!f)
            {
                fprintf(stderr, "%s: failed to open %s\n", __func__,
                        model_file.c_str());
                return false;
            }

            // seek past two uint32_t values in the beginning of the file
            // to skip the magic and hidden_size
            fseek(f, 2 * sizeof(uint32_t), SEEK_SET);

            for (;;)
            {
                // load all the weights from the file
                int32_t n_dims;
                int32_t length;

                fread(&n_dims, sizeof(int32_t), 1, f);
                fread(&length, sizeof(int32_t), 1, f);

                int32_t nelements = 1;
                int32_t ne[2] = {1, 1};
                for (int i = 0; i < n_dims; ++i)
                {
                    fread(&ne[i], sizeof(int32_t), 1, f);
                    nelements *= ne[i];
                }

                std::string name;
                std::vector<char> tmp(length);               // create a buffer
                fread(&tmp[0], sizeof(char), tmp.size(), f); // read to buffer
                name.assign(&tmp[0], tmp.size());

                // check if we reached eof of the open file f
                if (feof(f))
                {
                    break;
                }

                std::cout << "Loading tensor " << name << " with shape ["
                          << ne[0] << ", " << ne[1] << "]" << std::endl;

                // match the tensor name to the correct tensor in the model
                size_t loaded_size = 0;

                if (name == "input_mean")
                {
                    Eigen::MatrixXf mean_tmp = Eigen::MatrixXf(1487, 1);
                    loaded_size =
                        load_single_matrix(f, name, mean_tmp, ne, nelements);
                    // duplicate mean_tmp into model->input_mean[target_counter]
                    model->input_mean[target_counter].block(0, 0, 1487, 1) =
                        mean_tmp;
                    model->input_mean[target_counter].block(1487, 0, 1487, 1) =
                        mean_tmp;
                    model->input_mean[target_counter].transposeInPlace();
                }
                if (name == "input_scale")
                {
                    Eigen::MatrixXf scale_tmp = Eigen::MatrixXf(1487, 1);
                    loaded_size =
                        load_single_matrix(f, name, scale_tmp, ne, nelements);
                    // duplicate scale_tmp into
                    // model->input_scale[target_counter]
                    model->input_scale[target_counter].block(0, 0, 1487, 1) =
                        scale_tmp;
                    model->input_scale[target_counter].block(1487, 0, 1487, 1) =
                        scale_tmp;
                    model->input_scale[target_counter].transposeInPlace();
                }
                if (name == "output_mean")
                {
                    Eigen::MatrixXf mean_tmp = Eigen::MatrixXf(2049, 1);
                    loaded_size =
                        load_single_matrix(f, name, mean_tmp, ne, nelements);
                    // duplicate mean_tmp into
                    // model->output_mean[target_counter]
                    model->output_mean[target_counter].block(0, 0, 2049, 1) =
                        mean_tmp;
                    model->output_mean[target_counter].block(2049, 0, 2049, 1) =
                        mean_tmp;
                    model->output_mean[target_counter].transposeInPlace();
                }
                if (name == "output_scale")
                {
                    Eigen::MatrixXf scale_tmp = Eigen::MatrixXf(2049, 1);
                    loaded_size =
                        load_single_matrix(f, name, scale_tmp, ne, nelements);
                    // duplicate scale_tmp into
                    // model->output_scale[target_counter]
                    model->output_scale[target_counter].block(0, 0, 2049, 1) =
                        scale_tmp;
                    model->output_scale[target_counter].block(2049, 0, 2049,
                                                              1) = scale_tmp;
                    model->output_scale[target_counter].transposeInPlace();
                }
                if (name == "fc1.weight")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->fc1_w[target_counter], ne, nelements);
                }
                if (name == "bn1.weight")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn1_w[target_counter], ne, nelements);
                    model->bn1_w[target_counter].transposeInPlace();
                }
                if (name == "bn1.bias")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn1_b[target_counter], ne, nelements);
                    model->bn1_b[target_counter].transposeInPlace();
                }
                if (name == "bn1.running_mean")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn1_rm[target_counter], ne, nelements);
                    model->bn1_rm[target_counter].transposeInPlace();
                }
                if (name == "bn1.running_var")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn1_rv[target_counter], ne, nelements);
                    model->bn1_rv[target_counter].transposeInPlace();
                }
                if (name == "lstm.weight_ih_l0")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_w[target_counter][0][0], ne,
                        nelements);
                }
                if (name == "lstm.weight_hh_l0")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_w[target_counter][0][0], ne,
                        nelements);
                }
                if (name == "lstm.bias_ih_l0")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_b[target_counter][0][0], ne,
                        nelements);
                }
                if (name == "lstm.bias_hh_l0")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_b[target_counter][0][0], ne,
                        nelements);
                }
                if (name == "lstm.weight_ih_l0_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_w[target_counter][0][1], ne,
                        nelements);
                }
                if (name == "lstm.weight_hh_l0_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_w[target_counter][0][1], ne,
                        nelements);
                }
                if (name == "lstm.bias_ih_l0_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_b[target_counter][0][1], ne,
                        nelements);
                }
                if (name == "lstm.bias_hh_l0_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_b[target_counter][0][1], ne,
                        nelements);
                }
                if (name == "lstm.weight_ih_l1")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_w[target_counter][1][0], ne,
                        nelements);
                }
                if (name == "lstm.weight_hh_l1")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_w[target_counter][1][0], ne,
                        nelements);
                }
                if (name == "lstm.bias_ih_l1")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_b[target_counter][1][0], ne,
                        nelements);
                }
                if (name == "lstm.bias_hh_l1")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_b[target_counter][1][0], ne,
                        nelements);
                }
                if (name == "lstm.weight_ih_l1_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_w[target_counter][1][1], ne,
                        nelements);
                }
                if (name == "lstm.weight_hh_l1_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_w[target_counter][1][1], ne,
                        nelements);
                }
                if (name == "lstm.bias_ih_l1_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_b[target_counter][1][1], ne,
                        nelements);
                }
                if (name == "lstm.bias_hh_l1_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_b[target_counter][1][1], ne,
                        nelements);
                }
                if (name == "lstm.weight_ih_l2")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_w[target_counter][2][0], ne,
                        nelements);
                }
                if (name == "lstm.weight_hh_l2")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_w[target_counter][2][0], ne,
                        nelements);
                }
                if (name == "lstm.bias_ih_l2")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_b[target_counter][2][0], ne,
                        nelements);
                }
                if (name == "lstm.bias_hh_l2")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_b[target_counter][2][0], ne,
                        nelements);
                }
                if (name == "lstm.weight_ih_l2_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_w[target_counter][2][1], ne,
                        nelements);
                }
                if (name == "lstm.weight_hh_l2_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_w[target_counter][2][1], ne,
                        nelements);
                }
                if (name == "lstm.bias_ih_l2_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_ih_b[target_counter][2][1], ne,
                        nelements);
                }
                if (name == "lstm.bias_hh_l2_reverse")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->lstm_hh_b[target_counter][2][1], ne,
                        nelements);
                }
                if (name == "fc2.weight")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->fc2_w[target_counter], ne, nelements);
                }
                if (name == "bn2.weight")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn2_w[target_counter], ne, nelements);
                    model->bn2_w[target_counter].transposeInPlace();
                }
                if (name == "bn2.bias")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn2_b[target_counter], ne, nelements);
                    model->bn2_b[target_counter].transposeInPlace();
                }
                if (name == "bn2.running_mean")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn2_rm[target_counter], ne, nelements);
                    model->bn2_rm[target_counter].transposeInPlace();
                }
                if (name == "bn2.running_var")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn2_rv[target_counter], ne, nelements);
                    model->bn2_rv[target_counter].transposeInPlace();
                }
                if (name == "fc3.weight")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->fc3_w[target_counter], ne, nelements);
                }
                if (name == "bn3.weight")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn3_w[target_counter], ne, nelements);
                    model->bn3_w[target_counter].transposeInPlace();
                }
                if (name == "bn3.bias")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn3_b[target_counter], ne, nelements);
                    model->bn3_b[target_counter].transposeInPlace();
                }
                if (name == "bn3.running_mean")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn3_rm[target_counter], ne, nelements);
                    model->bn3_rm[target_counter].transposeInPlace();
                }
                if (name == "bn3.running_var")
                {
                    loaded_size = load_single_matrix(
                        f, name, model->bn3_rv[target_counter], ne, nelements);
                    model->bn3_rv[target_counter].transposeInPlace();
                }

                if (loaded_size == 0)
                {
                    printf("name is: '%s'\n", name.c_str());
                    fprintf(stderr, "%s: failed to load %s\n", __func__,
                            name.c_str());
                    return false;
                }
                total_size += loaded_size;
                n_loaded++;
            }

            fclose(f);
            target_counter += 1;
        }
    }

    // compute finish time in microseconds using std::chrono

    const auto t_end_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    // print load time in seconds
    printf("Loaded model (%u tensors, %6.2f MB) in %f s\n", n_loaded,
           total_size / 1024.0 / 1024.0,
           (float)(t_end_us - t_start_us) / 1000000.0f);

    return true;
}

// write a variant of load_single_tensor called load_single_matrix
// that takes an Eigen::MatrixXf &matrix and populates it from a file
static size_t load_single_matrix(FILE *f, std::string &name,
                                 Eigen::MatrixXf &matrix, int ne[2],
                                 int32_t nelements)
{
    if (matrix.size() != nelements ||
        (matrix.rows() != ne[0] || matrix.cols() != ne[1]))
    {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                __func__, name.data());
        fprintf(stderr,
                "%s: model file shape: [%d, %d], umx.cpp shape: [%d, %d]\n",
                __func__, ne[0], ne[1], (int)matrix.rows(), (int)matrix.cols());
        return 0;
    }

    const size_t bpe = sizeof(float);
    auto nbytes_tensor = matrix.size() * bpe;

    if ((nelements * bpe) != nbytes_tensor)
    {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.data(), nbytes_tensor, nelements * bpe);
        return 0;
    }

    fread(matrix.data(), bpe, nelements, f);

    printf("%16s: [%5d, %5d], type = float, %6.2f MB\n", name.data(), ne[0],
           ne[1], nbytes_tensor / 1024.0 / 1024.0);

    return nbytes_tensor;
}
