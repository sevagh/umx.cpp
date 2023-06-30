# umx.cpp

C++17 implementation of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (UMX), a PyTorch neural network for music demixing.

It uses [libnyquist](https://github.com/ddiakopoulos/libnyquist) to load audio files, the [ggml](https://github.com/ggerganov/ggml) file format to serialize the PyTorch weights of `umxhq` and `umxl` to a binary file format, and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (+ OpenMP) to implement the inference of Open-Unmix.

## Performance

The demixed output wav files (and their SDR score) of the main program [`umx.cpp`](./umx.cpp) are mostly identical to the PyTorch models (with the post-processing Wiener-EM step disabled):
```
# first, standard pytorch inference (no wiener-em)
$ python ./scripts/umx_pytorch_inference.py \
    --model=umxl \
    --dest-dir=./umx-py-xl-out \
    "/MUSDB18-HQ/test/Punkdisco - Oral Hygiene"

# then, inference with umx.cpp
$ umx.cpp.main ./ggml-umxl \
    "/MUSDB18-HQ/test/Punkdisco - Oral Hygiene" \
    ./umx-cpp-xl-out

# evaluate both, same SDR score

$ python ./scripts/evaluate-demixed-output.py \
    --musdb-root="/MUSDB18-HQ" \
    ./umx-py-xl-out \
    'Punkdisco - Oral Hygiene'

vocals          ==> SDR:   7.377  SIR:  16.028  ISR:  15.628  SAR:   8.376
drums           ==> SDR:   8.086  SIR:  12.205  ISR:  17.904  SAR:   9.055
bass            ==> SDR:   5.459  SIR:   8.830  ISR:  13.361  SAR:  10.543
other           ==> SDR:   1.442  SIR:   1.144  ISR:   5.199  SAR:   2.842

$ python ./scripts/evaluate-demixed-output.py \
    --musdb-root="/MUSDB18-HQ" \
    ./umx-cpp-xl-out \
    'Punkdisco - Oral Hygiene'

vocals          ==> SDR:   7.377  SIR:  16.028  ISR:  15.628  SAR:   8.376
drums           ==> SDR:   8.086  SIR:  12.205  ISR:  17.904  SAR:   9.055
bass            ==> SDR:   5.459  SIR:   8.830  ISR:  13.361  SAR:  10.543
other           ==> SDR:   1.442  SIR:   1.144  ISR:   5.199  SAR:   2.842
```

In runtime, this is actually slower than the PyTorch inference (and probably much slower than a possible Torch C++ inference implementation). For a 4:23 song, PyTorch takes 13s and umx.cpp takes 22s.

## Motivation

During the recent LLM hype (GPT4, ChatGPT, LLama, etc.), [ggerganov](https://github.com/ggerganov)'s projects [llama.cpp](https://github.com/ggerganov/llama.cpp) and a previous [whisper.cpp](https://github.com/ggerganov/whisper.cpp) gained popularity. These projects load the pretrained weights of the underlying neural network (trained using PyTorch) and reimplement the computations needed for inference in C without using Torch.

In the past, I've worked on a few derivations of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch), an open-source standard model for music demixing. The most recent model [UMX-L](https://zenodo.org/record/5069601) achieved a higher score than [UMX-HQ](https://zenodo.org/record/3370489).

I wanted to imitate llama.cpp and whisper.cpp on a neural network that I was familiar with, so I chose UMX. UMX is a small model (136 MB for UMX-HQ, 432 MB for UMX-L), so this is more of a technical curiosity than a necessity.

## Instructions

0. Clone the repo

Make sure you clone with submodules:
```
$ git clone --recurse-submodules https://github.com/sevagh/umx.cpp
```

1. Set up Python

The first step is to create a Python environment (however you like; I'm a fan of [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)) and install the `requirements.txt` file:
```
$ mamba create --name umxcpp python=3.10
$ mamba activate umxcpp
$ python -m pip install -r ./scripts/requirements.txt
```

2. Dump Open-Unmix weights to ggml files (use argument `--model=umxl`, `--model=umxhq` to switch between the two best pretrained models):
```
$ python ./scripts/convert-pth-to-ggml.py --model=umxl ./ggml-umxl
...
Skipping layer bn2.num_batches_tracked
Processing variable:  fc3.weight  with shape:  (4098, 1024)
Processing variable:  bn3.weight  with shape:  (4098,)
Processing variable:  bn3.bias  with shape:  (4098,)
Processing variable:  bn3.running_mean  with shape:  (4098,)
Processing variable:  bn3.running_var  with shape:  (4098,)
Skipping layer bn3.num_batches_tracked
Done. Output file:  ggml-umxl/ggml-model-umxl-other-f32.bin

```

This will load the model using PyTorch Torchhub (which implicitly downloads the weights files to the hidden torchhub folder), locate the weights files, and dump them using the [ggml](http://ggml.ai/) file format:
```
$ ls -latrh ggml-umxl/
total 432M
drwxrwxr-x  2 sevagh sevagh 4.0K Jun 28 10:14 .
drwxrwxr-x 13 sevagh sevagh 4.0K Jun 30 10:57 ..
-rw-rw-r--  1 sevagh sevagh 108M Jun 30 11:06 ggml-model-umxl-vocals-f32.bin
-rw-rw-r--  1 sevagh sevagh 108M Jun 30 11:06 ggml-model-umxl-drums-f32.bin
-rw-rw-r--  1 sevagh sevagh 108M Jun 30 11:06 ggml-model-umxl-bass-f32.bin
-rw-rw-r--  1 sevagh sevagh 108M Jun 30 11:06 ggml-model-umxl-other-f32.bin
```

3. Install C++ dependencies, e.g. CMake, gcc, C++/g++, Eigen, OpenMP for your OS - my instructions are for Pop!\_OS 22.04:
```
$ sudo apt-get install gcc g++ cmake clang-tools libeigen3-dev
```

4. Compile umx.cpp.main with CMake:
```
$ mkdir -p build && cd build && cmake .. && \
    make umx.cpp.main
```

Note: I have only tested this on my Linux-based computer (Pop!\_OS 22.04), and you may need to figure out how to get the dependencies on your own.

5. Run umx.cpp.main:
```
$ ./umx.cpp.main
Usage: ./umx.cpp.main <model dir> <wav file> <out dir>
umx.cpp Main driver program
Number of physical cores: 32
Input Samples: 23222488
Length in seconds: 263.294
Number of channels: 2
load_umx_model: loading model
Discovered model file "../ggml-umxl/ggml-model-umxl-other-f32.bin" in model dir../ggml-umxl/
Discovered model file "../ggml-umxl/ggml-model-umxl-drums-f32.bin" in model dir../ggml-umxl/
Discovered model file "../ggml-umxl/ggml-model-umxl-vocals-f32.bin" in model dir../ggml-umxl/
Discovered model file "../ggml-umxl/ggml-model-umxl-bass-f32.bin" in model dir../ggml-umxl/
Checking the magic of model_file ../ggml-umxl/ggml-model-umxl-bass-f32.bin
Checking the magic of model_file ../ggml-umxl/ggml-model-umxl-drums-f32.bin
Checking the magic of model_file ../ggml-umxl/ggml-model-umxl-other-f32.bin
Checking the magic of model_file ../ggml-umxl/ggml-model-umxl-vocals-f32.bin
Loaded umx model with hidden size 1024
Loading weights from model_file ../ggml-umxl/ggml-model-umxl-bass-f32.bin into target 0
Loading tensor input_mean with shape [1487, 1]
      input_mean: [ 1487,     1], type = float,   0.01 MB
Loading tensor input_scale with shape [1487, 1]
     input_scale: [ 1487,     1], type = float,   0.01 MB
Loading tensor output_scale with shape [2049, 1]
    output_scale: [ 2049,     1], type = float,   0.01 MB
Loading tensor output_mean with shape [2049, 1]
     output_mean: [ 2049,     1], type = float,   0.01 MB
Loading tensor fc1.weight with shape [2974, 1024]
      fc1.weight: [ 2974,  1024], type = float,  11.62 MB
Loading tensor bn1.weight with shape [1024, 1]
      bn1.weight: [ 1024,     1], type = float,   0.00 MB
Loading tensor bn1.bias with shape [1024, 1]
        bn1.bias: [ 1024,     1], type = float,   0.00 MB
Loading tensor bn1.running_mean with shape [1024, 1]
bn1.running_mean: [ 1024,     1], type = float,   0.00 MB
Loading tensor bn1.running_var with shape [1024, 1]

... <truncated>

Loaded model (172 tensors, 431.36 MB) in 0.131382 s
umx_model_load returned true
Computing STFT
spec shape: (incl 2 chan) 11340 x 2049
Computing STFT magnitude
Computing STFT phase
Running inference with Eigen matrices

Writing wav file "./aam-out/target_0.wav" to ./aam-out
Encoder Status: 0
Writing wav file "./aam-out/target_2.wav" to ./aam-out
Encoder Status: 0
Writing wav file "./aam-out/target_1.wav" to ./aam-out
Encoder Status: 0
Writing wav file "./aam-out/target_3.wav" to ./aam-out
Encoder Status: 0
```

## Design

I took the following steps to write umx.cpp:

1. Create STFT/iSTFT functions equivalent to the PyTorch stft used by Open-Unmix

The source file [dsp.cpp](./src/dsp.cpp) contains STFT/iSTFT functions with center padding and window scaling using Eigen's `unsupported/Eigen/FFT` (which is more or less the same as KissFFT, adapted to use the C++ standard library types).

The script [compare-torch-stft.py](./scripts/compare-torch-stft.py) uses `openunmix.transforms.make_filterbanks` to return the same STFT/iSTFT used in UMX inference (and print some values for debugging), and from [test_dsp.cpp](./test/test_dsp.cpp), I was able to print the same values until I obtained the same outputs.

2. Create supporting functions (load audio waveform, magnitude/phase spectrograms, and getting a complex spectrogram from the polar form)

All can be seen in dsp.cpp.

3. Write [convert-pth-to-ggml.py](./scripts/convert-pth-to-ggml.py), which is borrowed from whisper.cpp.

This rather straightforwardly loads each PyTorch weight tensor and dumps them to a binary file.

4. Write [model.cpp](./src/model.cpp), which is also borrowed from whisper.cpp, and loads the binary files into `Eigen::MatrixXf` weight matrices

5. Implement the forward inference operations using the weight matrices in [umx.cpp](./umx.cpp), with the more complex LSTM code in [lstm.cpp](./src/lstm.cpp)

This was done by reading the PyTorch documentation for each module, writing the equations using Eigen, and printing the outputs of each layer in PyTorch and umx.cpp until bugs were fixed and the outputs were identical.

### Layer implementations

**Input/output scale + mean**

PyTorch:
```
x = x*self.input_scale + self.input_mean
```

C++ with Eigen:
```
// clone input mix mag x
Eigen::MatrixXf x_input = x;

// apply formula x = x*input_scale + input_mean
#pragma omp parallel for
for (int i = 0; i < x_input.rows(); i++)
{
    x_input.row(i) = x_input.row(i).array() *
                              model.input_scale.array() +
                          model.input_mean.array();
}
```

**Fully-connected/linear layers (with no bias)**

PyTorch:
```
x = self.fc1(x)
```

C++ with Eigen:
```
// y = x A^T + b
// x = (nb_frames, in_features)
// A = weights = (out_features, in_features)
// A^T = A transpose = (in_features, out_features)
x_input = x_input * model.fc1_w;
```

**Batchnorm1d**

PyTorch:
```
x = self.bn1(x)
```

C++ with Eigen:
```
// batchnorm1d calculation
// y=(x-E[x])/(sqrt(Var[x]+ϵ) * gamma + Beta
#pragma omp parallel for
    for (int i = 0; i < x_input.rows(); i++)
    {
        x_input.row(i) =
            (((x_input.row(i).array() -
               model.bn1_rm.array()) /
              (model.bn1_rv.array() + 1e-5).sqrt()) *
                 model.bn1_w.array() +
             model.bn1_b.array())
                .tanh();
    }
```

**LSTM (multilayer bidirectional)**

PyTorch:
```
lstm_out = self.lstm(x)
```

C++ with Eigen:
```
// create Zero matrices for hidden, cell states

Eigen::MatrixXf loop_input = input;

for (int lstm_layer = 0; lstm_layer < 3; ++lstm_layer)
{
// parallelize the directions which don't depend on each other
#pragma omp parallel for
    for (int direction = 0; direction < 2; ++direction)
    {
        // forward direction = 0: for t = 0 to seq_len - 1
        // backward direction = 1: for t = seq_len - 1 to 0
        for (int t = (direction == 0 ? 0 : seq_len - 1);
             (direction == 0 ? t < seq_len : t > -1);
             t += (direction == 0 ? 1 : -1))
        {
            // apply the inner input/hidden gate calculation for all gates
            // W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh
            //
            // at the end of the loop iteration, h[lstm_layer][direction]
            // will store h_t of this iteration at the beginning of the next
            // loop iteration, h[lstm_layer][direction] will be h_{t-1},
            // which is what we want similar for c[lstm_layer][direction]
            // and c_{t-1}
            //
            // the initial values for h and c are 0
            Eigen::MatrixXf gates =
                model.lstm_ih_w[lstm_layer][direction].transpose() *
                    loop_input.row(t).transpose() +
                model.lstm_ih_b[lstm_layer][direction] +
                model.lstm_hh_w[lstm_layer][direction].transpose() *
                    h[lstm_layer][direction] +
                model.lstm_hh_b[lstm_layer][direction];

            // slice up the gates into i|f|g|o-sized chunks
            Eigen::MatrixXf i_t =
                sigmoid(gates.block(0, 0, hidden_state_size, 1));
            Eigen::MatrixXf f_t = sigmoid(
                gates.block(hidden_state_size, 0, hidden_state_size, 1));
            Eigen::MatrixXf g_t = (gates.block(2 * hidden_state_size, 0,
                                               hidden_state_size, 1))
                                      .array()
                                      .tanh();
            Eigen::MatrixXf o_t = sigmoid(gates.block(
                3 * hidden_state_size, 0, hidden_state_size, 1));

            Eigen::MatrixXf c_t =
                f_t.array() * c[lstm_layer][direction].array() +
                i_t.array() * g_t.array();
            Eigen::MatrixXf h_t = o_t.array() * (c_t.array().tanh());

            // store the hidden and cell states for later use
            h[lstm_layer][direction] = h_t;
            c[lstm_layer][direction] = c_t;

            output_per_direction[lstm_layer][direction].row(t)
                << h_t.transpose();
        }
    }

    // after both directions are done per LSTM layer, concatenate the
    // outputs
    output[lstm_layer] << output_per_direction[lstm_layer][0],
        output_per_direction[lstm_layer][1];

    loop_input = output[lstm_layer];
}

// return the concatenated forward and backward hidden state as the final
// output
return output[2];
```
