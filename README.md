# umx.cpp

C++17 implementation of [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (UMX), a PyTorch neural network for music demixing. It uses [libnyquist](https://github.com/ddiakopoulos/libnyquist) to load audio files, the [ggml](https://github.com/ggerganov/ggml) file format to serialize the PyTorch weights of `umxhq` to a binary file format, and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to implement the inference of Open-Unmix.

There are 3 main differences in umx.cpp that deviate from the PyTorch model:
* **Quantized and compressed weights:** the best-performing [UMX-L](https://zenodo.org/record/5069601) weights are quantized (mostly uint8, uint16 for the final four layers) and saved with the [ggml](https://github.com/ggerganov/ggml) binary file format and then gzipped. This reduces the 425 MB of UMX-L weights down to 45 MB, while achieving similar performance (verified empirically using BSS metrics)
* **Segmented inference:** we borrow the overlapping segmented inference from [Demucs](https://github.com/facebookresearch/demucs/blob/main/demucs/apply.py#L264-L297) (and in turn [demucs.cpp](https://github.com/sevagh/demucs.cpp/blob/21e76ca781c4411bef073ace06d8e84c3c5c9835/src/model_apply.cpp#L180-L263)), which is very effective at processing a waveform in small chunks while avoiding discontinuities at the left and right boundaries when recombined with its neighboring chunks
* **Streaming LSTM:** following the above, since we chunk the input waveform, we can adapt the LSTM such that it's temporal sequence length is the chunk length, and each chunk is _streamed_ through the LSTM; again, we verified empirically with BSS metrics that this resulted in a similar overall SDR score while reducing memory and computation footprints

## Open-Unmix (UMX-L)

MUSDB18-HQ test track 'Zeno - Signs':

'Zeno - Signs', fully segmented (60s) inference + wiener + streaming lstm + uint8/16-quantized gzipped model file:
```
vocals          ==> SDR:   6.836  SIR:  16.416  ISR:  14.015  SAR:   7.065
drums           ==> SDR:   7.434  SIR:  14.580  ISR:  12.057  SAR:   8.906
bass            ==> SDR:   2.445  SIR:   4.817  ISR:   5.349  SAR:   3.623
other           ==> SDR:   6.234  SIR:   9.421  ISR:  12.515  SAR:   7.611
```

'Zeno - Signs', fully segmented (60s) inference + wiener + streaming lstm, no uint8 quantization:
```
vocals          ==> SDR:   6.830  SIR:  16.421  ISR:  14.044  SAR:   7.104
drums           ==> SDR:   7.425  SIR:  14.570  ISR:  12.062  SAR:   8.905
bass            ==> SDR:   2.462  SIR:   4.859  ISR:   5.346  SAR:   3.566
other           ==> SDR:   6.197  SIR:   9.437  ISR:  12.519  SAR:   7.627
```

'Zeno - Signs', unsegmented inference (crashes with large tracks) w/ streaming lstm + wiener:
```
vocals          ==> SDR:   6.846  SIR:  16.382  ISR:  13.897  SAR:   7.024
drums           ==> SDR:   7.679  SIR:  14.462  ISR:  12.606  SAR:   9.001
bass            ==> SDR:   2.386  SIR:   4.504  ISR:   5.802  SAR:   3.731
other           ==> SDR:   6.020  SIR:   9.854  ISR:  11.963  SAR:   7.472
```

Original release results on 'Zeno - Signs' (no streaming LSTM, no Wiener filtering):
```
vocals          ==> SDR:   6.550  SIR:  14.583  ISR:  13.820  SAR:   6.974
drums           ==> SDR:   6.538  SIR:  11.209  ISR:  11.163  SAR:   8.317
bass            ==> SDR:   1.646  SIR:   0.931  ISR:   5.261  SAR:   2.944
other           ==> SDR:   5.190  SIR:   6.623  ISR:  10.221  SAR:   8.599
```

* Streaming UMX LSTM module for longer tracks with Demucs overlapping segment inference

Testing 'Georgia Wonder - Siren' (largest MUSDB track) for memory usage with 60s segments:
```
vocals          ==> SDR:   5.858  SIR:  10.880  ISR:  14.336  SAR:   6.187
drums           ==> SDR:   7.654  SIR:  14.933  ISR:  11.459  SAR:   8.466
bass            ==> SDR:   7.256  SIR:  12.007  ISR:  10.743  SAR:   6.757
other           ==> SDR:   4.699  SIR:   7.452  ISR:   9.142  SAR:   4.298
```

vs. pytorch inference (w/ wiener):
```
vocals          ==> SDR:   5.899  SIR:  10.766  ISR:  14.348  SAR:   6.187
drums           ==> SDR:   7.939  SIR:  14.676  ISR:  12.485  SAR:   8.383
bass            ==> SDR:   7.576  SIR:  12.712  ISR:  11.188  SAR:   6.951
other           ==> SDR:   4.624  SIR:   7.937  ISR:   8.845  SAR:   4.270
```

## Performance

The demixed output wav files (and their SDR score) of the main program [`umx.cpp`](./umx.cpp) are mostly identical to the PyTorch models:
```
# first, standard pytorch inference
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

vocals          ==> SDR:   7.695  SIR:  17.312  ISR:  16.426  SAR:   8.322
drums           ==> SDR:   8.899  SIR:  14.054  ISR:  14.941  SAR:   9.428
bass            ==> SDR:   8.338  SIR:  14.352  ISR:  14.171  SAR:  10.971
other           ==> SDR:   2.017  SIR:   6.266  ISR:   6.821  SAR:   2.410

$ python ./scripts/evaluate-demixed-output.py \
    --musdb-root="/MUSDB18-HQ" \
    ./umx-cpp-xl-out \
    'Punkdisco - Oral Hygiene'

vocals          ==> SDR:   7.750  SIR:  17.510  ISR:  16.195  SAR:   8.321
drums           ==> SDR:   9.010  SIR:  14.149  ISR:  14.900  SAR:   9.416
bass            ==> SDR:   8.349  SIR:  14.348  ISR:  14.160  SAR:  10.990
other           ==> SDR:   1.987  SIR:   6.282  ISR:   6.674  SAR:   2.461
```

In runtime, this is actually slower than the PyTorch inference (and probably much slower than a possible Torch C++ inference implementation).

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

2. Dump Open-Unmix weights to ggml files (use argument `--model=umxl`, `--model=umxhq` to switch between the two best pretrained models)\*:
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
Done. Output file:  ggml-models/ggml-model-umxl-u8.bin
```
\*: :warning: my script can no longer find `umxhq` files on Zenodo, so `umxl` is the new default

This will load the model using PyTorch Torchhub (which implicitly downloads the weights files to the hidden torchhub folder), locate the weights files, and dump them using the [ggml](http://ggml.ai/) file format with mixed uint8 and uint16 quantization, which you can then gzip:
```
# gzip in-place
$ gzip -k ./ggml-models/ggml-model-umxl-u8.bin
$ ls -latrh ggml-models/
total 177M
-rw-rw-r--  1 sevagh sevagh  45M Dec 30 08:25 ggml-model-umxl-u8.bin.gz
drwxrwxr-x 13 sevagh sevagh 4.0K Dec 30 09:13 ..
drwxrwxr-x  2 sevagh sevagh 4.0K Dec 30 09:33 .
-rw-rw-r--  1 sevagh sevagh 132M Dec 30 09:33 ggml-model-umxl-u8.bin
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

$ ./umx.cpp.main ./ggml-umxl ./test.wav ./demix-out-umxl
umx.cpp Main driver program
Number of physical cores: 32
Input Samples: 20672662
Length in seconds: 234.384
Number of channels: 2
load_umx_model: loading model
Decompressing model_file... ../ggml-models/ggml-model-umxl-u8.bin.gz
Checking the magic of model_file ../ggml-models/ggml-model-umxl-u8.bin.gz
Loaded umx model with hidden size 1024
Loading weights from model_file ../ggml-models/ggml-model-umxl-u8.bin.gz
Loading target 0
Loading tensor input_mean with shape [1487, 1]
      input_mean: [ 1487,     1], type = float,   0.00 MB
Loading target 0
... <truncated>
Loaded model (172 tensors, 131.93 MB) in 1.271085 s
umx_model_load returned true
Per-segment progress: 0.166667
2., apply model w/ split, offset: 0, chunk shape: (2, 2646000)
Generating spectrograms
populate eigen matrixxf
Input scaling
Target 0 fc1
Target 0 bn1
Target 0 lstm
Target 0 fc2
Target 0 bn2
Target 0 fc3
Target 0 bn3
Target 0 output scaling
Multiply mix mag with computed mask
Multiply mix mag with computed mask
... <truncated>
Getting complex spec from wiener filtering
Wiener-EM: Getting first estimates from naive mix-phase
Wiener-EM: Scaling down by max_abs
Wiener-EM: Initialize tensors
... <truncated>
Getting waveforms from istft
Writing wav file Writing wav file Writing wav file "./umx-cpp-out/target_2.wav""./umx-cpp-out/target_3.wav" to ./umx-cpp-out
 to ./umx-cpp-out
"./umx-cpp-out/target_1.wav" to ./umx-cpp-out
Writing wav file "./umx-cpp-out/target_0.wav" to ./umx-cpp-out
Encoder Status: 0
Encoder Status: 0
Encoder Status: 0
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
