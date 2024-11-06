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

### Training the Models

To train any of these models, use the following script:

```bash
bash run_training.sh

