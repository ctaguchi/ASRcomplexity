# Datasets
|Language|Dataset|Phonemizer|
|--------|-------|----------|
|Japanese|[SNOW](https://huggingface.co/datasets/snow_simplified_japanese_corpus)|[SudachiPy](https://github.com/WorksApplications/SudachiPy), [pykakasi](https://pykakasi.readthedocs.io/)|
|Chinese |[LCCC](https://huggingface.co/datasets/lccc)				 |[dragonmapper](https://github.com/tsroten/dragonmapper)|
|Korean	 |[Korean Parallel Corpora](https://huggingface.co/datasets/Moo/korean-parallel-corpora)|[g2pK](https://github.com/Kyubyong/g2pK)|
|Thai	 |[ThaiGov V2 Corpus](https://huggingface.co/datasets/pythainlp/thaigov-v2-corpus-22032023)|[PyThaiNLP](https://pythainlp.github.io)|
|Arabic	 |[Rasaif Classical-Arabic-English Parallel Texts](https://huggingface.co/datasets/ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts)|[gruut](https://github.com/rhasspy/gruut)|
|English |[Europarl](https://www.statmt.org/europarl)|gruut|
|French	 |Europarl|gruut|
|Italian |Europarl|gruut|
|Czech	 |Europarl|gruut|
|Swedish |Europarl|gruut|
|Dutch	 |Europarl|gruut|
|German	 |Europarl|gruut|

## Notes on the experiments
Most of the experimental setup follows the original work by Sproat & Gutkin (2021).
Following their work, the implementation code was initially based on [Tensorflow's NMT tutorial](https://www.tensorflow.org/text/tutorials/nmt_with_attention).
The different settings specific to this experiment are:
- 8000 samples and 2000 samples were used for the training and validation data, respectively.
- The training was run for 10 epochs.