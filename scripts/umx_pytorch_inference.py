#!/usr/bin/env python
import openunmix
from openunmix.filtering import wiener
import torch
import torchaudio.backend.sox_io_backend
import torchaudio
import argparse
import numpy as np
import os

# sorted order of ggml bin file names
target_digit_map = {
    'bass': 0,
    'drums': 1,
    'other': 2,
    'vocals': 3,
}


if __name__ == '__main__':
    # set up argparse with input wav file as positional argument
    parser = argparse.ArgumentParser(description='Open Unmix - Audio Source Separation')
    parser.add_argument('input_file', type=str, help='path to input wav file')
    parser.add_argument('--dest-dir', type=str, default=None, help='path to write output files')
    parser.add_argument('--model', type=str, default='umxhq', help='(umxhq, umxl)')

    args = parser.parse_args()

    # load audio file and resample to 44100 Hz
    metadata = torchaudio.info(args.input_file)
    print(metadata)
    audio, rate = torchaudio.load(args.input_file)

    # Load model
    umx_module = ''
    if args.model == 'umxhq':
        umx_module = 'openunmix.umxhq_spec'
    elif args.model == 'umxl':
        umx_module = 'openunmix.umxl_spec'

    model = eval(umx_module)()

    # Perform inference with spectrogram
    stft, istft = openunmix.transforms.make_filterbanks(n_fft=4096, n_hop=1024, center=True, sample_rate=44100.0, method="torch")
    audio = torch.unsqueeze(audio, dim=0)
    spec = stft(audio)
    mag_spec = torch.abs(torch.view_as_complex(spec))
    phase_spec = torch.angle(torch.view_as_complex(spec))

    out_mag_specs = []

    # UMX forward inference
    for target_name, target_model in model.items():
        print(f"Inference for target {target_name}")
        out_mag_spec = target_model(mag_spec)
        print(type(out_mag_spec))
        print(out_mag_spec.shape)
        out_mag_specs.append(torch.unsqueeze(out_mag_spec, dim=-1))

    out_mag_spec_concat = torch.cat(out_mag_specs, dim=-1)
    print(f"shape, dtype: {out_mag_spec_concat.shape}, {out_mag_spec_concat.dtype}")

    # Convert back to complex tensor
    #out_spec = out_mag_spec * torch.exp(1j * phase_spec)
    # do wiener filtering

    wiener_mag_inp = out_mag_spec_concat[0, ...].permute(2, 1, 0, 3)
    wiener_spec_inp = spec[0, ...].permute(2, 1, 0, 3)

    out_specs = wiener(wiener_mag_inp, wiener_spec_inp)

    # out_specs: torch.Size([44, 2049, 2, 2, 4])
    # nb_frames, nb_bins, nb_channels, 2, targets
    # 0          1        2            3  4
    # permute:
    #           4  2  1  0  3

    # samples, targets, channels, nb_bins, nb_frames, 2
    out_specs = torch.unsqueeze(out_specs.permute(4, 2, 1, 0, 3), dim=0)
    out_audios = istft(out_specs)[0]
    print(out_audios.shape)

    # get istft
    for i, target_name in enumerate(model.keys()):
        # write to file in directory
        if args.dest_dir is not None:
            os.makedirs(args.dest_dir, exist_ok=True)
            torchaudio.save(os.path.join(args.dest_dir, f'target_{target_digit_map[target_name]}.wav'), out_audios[i], sample_rate=44100)

    print("Goodbye!")
