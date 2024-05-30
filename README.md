# Language Complexity and Speech Recognition Accuracy
This repository stores the final results and the code to reproduce the experiments.

- `main.py`: ASR training code.
- logographicity
  - `train.py`: Code for training an encoder-decoder model (with attention) for measuring logographicity
  - `logographicity_metrics.py`: Code for measuring logographicity scores.

# Results
|Language  |Family                         |Writing system|Script type          |CER (%)|\#Grapheme|Unigram entropy|Logographicity|\#Phoneme|
|----------|-------------------------------|--------------|---------------------|-------|----------|---------------|--------------|---------|
|Japanese  |Japonic                        |kanji+kana    |logographic+syllabary|78.63  |1702      |7.74           |44.27         |27       |
|Japanese  |Japonic                        |kana          |syllabary            |25.37  |92        |5.63           |              |         |
|Japanese  |Japonic                        |romaji        |alphabetic           |17.25  |27        |3.52           |              |         |
|Korean    |Koreanic                       |hangul        |syllabary            |30.12  |965       |7.98           |25.67         |39.5     |
|Korean    |Koreanic                       |jamo          |alphabetic           |18.25  |62        |4.90           |              |         |
|Chinese   |Sinitic, Sino-Tibetan          |hanzi         |logographic          |75.79  |2155      |9.47           |41.59         |42.5     |
|Chinese   |Sinitic, Sino-Tibetan     	   |zhuyin     	  |semisyllabary        |13.65  |49        |4.81           |              |         |
|Chinese   |Sinitic, Sino-Tibetan          |pinyin        |alphabetic      	|12.49	|56	   |5.02	   |		  |	    |
|Thai      |Tai, Kra-Dai                   |Thai          |Abugida		|24.00	|67	   |5.24	   |20.55	  |40.67    |
|Arabic    |Semitic, Afroasiatic           |Perso-Arabic  |Abjad		|42.72	|53	   |4.77	   |21.57	  |37	    |
|English   |Germanic, Indo-European        |Latin         |alphabetic		|4.36	|27	   |4.17	   |19.17	  |41.22    |
|French    |Romance, Indo-European         |Latin         |alphabetic		|22.05	|69	   |4.42	   |20.37	  |36.75    |
|Italian   |Romance, Indo-European         |Latin         |alphabetic		|16.63	|48	   |4.27	   |21.28	  |43.33    |
|Czech     |Slavic, Indo-European          |Latin         |alphabetic		|19.19	|46	   |4.92	   |20.57	  |39	    |
|Swedish   |Germanic, Indo-European        |Latin         |alphabetic		|23.17	|34	   |41.52	   |19.81	  |35	    |
|Dutch     |Germanic, Indo-European        |Latin     	  |alphabetic		|14.20	|36	   |4.20	   |16.67	  |49.38    |
|German    |Germanic, Indo-European        |Latin     	  |alphabetic		|9.43	|48	   |4.18	   |18.03	  |40	    |
|Lithuanian|Baltic, Indo-European          |Latin     	  |alphabetic		|14.88	|40	   |4.56	   |		  |52.5	    |
|Polish    |Slavic, Indo-European          |Latin     	  |alphabetic		|14.88	|40	   |4.56	   |		  |36	    |
|Basque    |Basque                         |Latin     	  |alphabetic		|8.78	|27	   |3.89	   |		  |30.71    |
|Indonesian|Malayo-Polynesian, Austronesian|Latin     	  |alphabetic		|26.63	|35	   |4.04	   |		  |31	    |
|Kabyle    |Berber, Afroasiatic            |Latin     	  |alphabetic		|35.87	|46	   |4.30	   |		  |30	    |
|Swahili   |Bantu, Niger-Congo             |Latin     	  |alphabetic		|19.92	|33	   |4.00	   |		  |36.5	    |
|Hungarian |Ugric, Uralic                  |Latin     	  |alphabetic		|17.22	|37	   |4.52	   |		  |52	    |
|Russian   |Slavic, Indo-European          |Cyrillic      |alphabetic		|16.47	|40	   |4.65	   |		  |39.33    |
|Tatar     |Kipchak, Turkic                |Cyrillic  	  |alphabetic		|24.18	|43	   |4.72	   |		  |43	    |
|Abkhaz    |Northwest Caucasian            |Cyrillic  	  |alphabetic		|16.60	|41	   |4.60	   |		  |66	    |
|Georgian  |Kartvelian                     |Georgian      |alphabetic		|17.04	|37	   |4.30	   |		  |33.75    |
|Armenian  |Armenian, Indo-European        |Armenian      |alphabetic		|13.88	|49	   |4.57	   |		  |36.5	    |
|Hindi     |Indo-Aryan, Indo-European      |Devanagari    |Abugida		|23.61	|119	   |5.10	   |		  |68.4	    |

## Notes on the experiments
- The numbers of graphemes and the unigram entropies were counted in the training and validation datasets.
- The logographicity scores are calculated only for languages that have available phonemizers.
- The logographicity scores are calculated based on separate datasets (i.e., not CommonVoice). See the README.md in the logographicity directory.
- For romaji Japanese, phonemic symbol /q/ was used to denote phonemic geminates.
- For pinyin Chinese, tonal symbols were not counted as separate characters, but each combination of a vowel and a tonal symbol was counted as a distinct character.
- For English, we did not use CommonVoice due to the uncertain quality of the accents of the contributors; instead, LibriSpeech was used.