<<<<<<< HEAD
# TPSE-VC
# 1. Preprocessing
The training and testing dataset is [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html). We also use some preprocess script from [ericwudayi](https://github.com/ericwudayi)/[SkipVQVC](https://github.com/ericwudayi/SkipVQVC) to generate acoustic features.
=======
# tpse-vc

We are cleaning the code, and will open source in the furture.
>>>>>>> b5402851506ccb9e2cd0bc8b2f9b0c78be2752f6



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
