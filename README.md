# TPSE-VC
# Preprocessing
The training and testing dataset is [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443). We also use some preprocess script from [ericwudayi](https://github.com/ericwudayi)/[SkipVQVC](https://github.com/ericwudayi/SkipVQVC) to generate acoustic features.

1. processing waveform to acoustic features.

   ```
   python data/preprocessing_vctk.py
   ```

2. splitting training and testing datasets.

   ```
   python data/split_vctk.py
   ```

   

# Training
```
python main.py
```

Some configurations depend on ```config.yaml```.

# Inference
You can use ```inference_utterance.py``` to inference, which depends on a json file.

```json
[["p239_043.wav", "p255_090.wav"], ["p249_169.wav", "p257_035.wav"], ..., ]
```

For each list, the first and the second ones indicate the source utterances and target utterances respectively.



# Acknowledge
Our implementation is hugely influenced by the following repositories as we benefit a lot from their codes and papers:

- [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion)
- [VQVC+](https://github.com/ericwudayi/SkipVQVC)
- [AutoVC](https://github.com/auspicious3000/autovc)
- [MelGAN](https://github.com/descriptinc/melgan-neurips) 
