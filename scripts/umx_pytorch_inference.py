#!/usr/bin/env python
import openunmix
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

    #model = eval(umx_module)()
    model = openunmix.umxl()

    # Perform inference with spectrogram
    #stft, istft = openunmix.transforms.make_filterbanks(n_fft=4096, n_hop=1024, center=True, sample_rate=44100.0, method="torch")
    #spec = stft(audio)
    #mag_spec = torch.abs(torch.view_as_complex(spec))
    #phase_spec = torch.angle(torch.view_as_complex(spec))

    #out_mag_specs = []

    ## UMX forward inference
    #for target_name, target_model in model.items():
    #    print(f"Inference for target {target_name}")
    #    out_mag_specs.append(torch.unsqueeze(target_model(mag_spec), dim=-1))

    #out_mag_spec = torch.cat(out_mag_specs, dim=-1)
    #print(out_mag_spec.shape)

    ## Convert back to complex tensor

    ## apply wiener filter
    #wiener_win_len = 300
    #pos = 0
    #nb_frames = out_mag_spec.shape[-2]

    #out_spec = torch.zeros((*spec.shape, 4,), dtype=torch.float32)

    #print(f"out_mag_spec shape: {out_mag_spec.shape}")
    #print(f"spec shape: {spec.shape}")

    #while pos < nb_frames:
    #    cur_frame = torch.arange(pos, min(nb_frames, pos+wiener_win_len))
    #    tmp = openunmix.filtering.wiener(
    #        out_mag_spec[0, :, :, cur_frame, :],
    #        spec[0, :, :, cur_frame, :],
    #        1,
    #        False,
    #        False
    #    )
    #    out_spec[0, :, :, cur_frame, :] = tmp.permute(2, 1, 0, 3, 4)

    #print(f"FINISH: {out_spec.shape}")

    audio = torch.unsqueeze(audio, dim=0)
    out = model(audio)
    print(f"out shape: {out.shape}")
    out = model.to_dict(out)
    #out = torch.squeeze(out, dim=0)

    for target_name, target_waveform in out.items():
        # get istft
        print(target_waveform.shape)
        out_audio = torch.squeeze(target_waveform, dim=0)
        print(f"writing target {target_name} to file {target_digit_map[target_name]}")

        # write to file in directory
        if args.dest_dir is not None:
            os.makedirs(args.dest_dir, exist_ok=True)
            torchaudio.save(os.path.join(args.dest_dir, f'target_{target_digit_map[target_name]}.wav'), out_audio, sample_rate=44100)

    print("Goodbye!")
