# ASR-CTC Models Repository

This repository contains implementations of three Automatic Speech Recognition (ASR) models based on Connectionist Temporal Classification (CTC):

1. **CTC** - A minimal implementation of the basic CTC model.  
   *Reference*: [Connectionist Temporal Classification](https://dl.acm.org/doi/10.1145/1143844.1143891)
   - Citation:  
     ```
     @inproceedings{graves2006connectionist,
       title={Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks},
       author={Graves, Alex and Fern{\'a}ndez, Santiago and Gomez, Faustino and Schmidhuber, J{\"u}rgen},
       booktitle={Proceedings of the 23rd international conference on Machine learning},
       pages={369--376},
       year={2006}
     }
     ```

2. **SC-CTC** - Self-Conditioned CTC, which introduces self-conditioning to the CTC framework.  
   *Paper Reference*: [Relaxing the Conditional Independence Assumption of CTC-Based ASR](https://arxiv.org/abs/2104.02724)
   - Citation:  
     ```
     @article{nozaki2021relaxing,
       title={Relaxing the conditional independence assumption of CTC-based ASR by conditioning on intermediate predictions},
       author={Nozaki, Jumon and Komatsu, Tatsuya},
       journal={arXiv preprint arXiv:2104.02724},
       year={2021}
     }
     ```

3. **HC-CTC** - Hierarchically Conditioned CTC, which applies a hierarchical conditioning mechanism for improved performance.  
   *Paper Reference*: [Hierarchical Conditional End-to-End ASR with CTC](https://arxiv.org/abs/2110.04109)
   - Citation:  
     ```
     @inproceedings{higuchi2022hierarchical,
       title={Hierarchical conditional end-to-end asr with ctc and multi-granular subword units},
       author={Higuchi, Yosuke and Karube, Keita and Ogawa, Tetsuji and Kobayashi, Tetsunori},
       booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
       pages={7797--7801},
       year={2022},
       organization={IEEE}
     }
     ```
The convolutional subsampling approach is taken from [Fast Conformer Nvidia](https://arxiv.org/pdf/2305.05084) which uses an 8x downsampling of audio making the Conformer very fast.

However, a lower downsampling of 4x can be used by setting `cfg.features.downsample: 4`


`requirements.txt` contains the libraries I had installed (some of them might not be needed).

### Training the Models

To train any of these models, use the following script:

```bash
bash run_training.sh
```

### Configuration

The configs for all models can be found in `config/` directory.

### Decoding

The `run_decode.sh` will run decoding over a corpus and report WER

To decode a specific audio file use a `yaml` config `cfg`,
```
model = baseHCCTC.from_pretrained(cfg, cfg.paths.ckpt_path) # cfg.paths.ckpt_path path where checkpoint is saved
transcription = model.transcribe(/path/to/audio_file)
print(transcription)
```

### Results

Here are the performance WER on various datasets using a HC-CTC model trained for 48 epochs on a combination of datasets.
We also compare with Whisper-small which is about the same size as our model.

| Dataset                 | WER (no finetuning) | WER (after finetuning) | WER (Whisper-small) |
|-------------------------|---------------------|-------------------------|---------------------|
| librispeech test-clean  | 3.52                | 3.1                     | 3.2                 |
| librispeech test-other  | 7.89                | 7.05                    | 6.7                 |
| switchboard             | 11.2                | 10.7                    | 13.4                |
| callhome                | 18.17               | 17.56                   | 17.2                |
| slurp                   | 13.86               | 12.11                   | 17.8                |
| tedlium2                | 6.48                | 6.11                    | 6.49                |

