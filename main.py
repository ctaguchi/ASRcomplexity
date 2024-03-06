from datasets import (load_dataset,
                      Dataset,
                      Audio,
                      concatenate_datasets,
                      IterableDataset,
                      load_from_disk)
import pandas as pd
import json
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor,
                          Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer,
                          WhisperFeatureExtractor,
                          WhisperTokenizer,
                          WhisperProcessor,
                          WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import time
from argparse import ArgumentParser
import os
from functools import partial
import evaluate
import numpy as np
import sys
import re
import string

# For the converter
import pykakasi
import sudachipy.tokenizer
import sudachipy.dictionary
import dragonmapper.hanzi
import translit_tt
from pythainlp.transliterate import transliterate

# Local
import hangul
from unigram_entropy import UnigramEntropy

DATASET_REPO = "mozilla-foundation/common_voice_16_1" # newest as of Jan 19, 2024
ZEROTH_KOREAN = "Bingsu/zeroth-korean"
LIBRISPEECH = "librispeech_asr"

class DataLoader:
    def __init__(self, lang, max_sample=10_000, max_audio_len=15):
        self.lang = lang
        self.max_sample = max_sample
        self.max_audio_len = max_audio_len

    def remove_punctuation(self, batch: dict) -> dict:
        chars_to_ignore_regex = "[\,\?\.\!\-\;\:\"]"
        japanese_punct_regex = "[、。；：？！（）＞＜，．＠『』「」＝〜％＆＃＋＊]"
        s = batch["sentence"]
        s = s.translate(str.maketrans("", "", string.punctuation)) # remove English punctuation
        s = re.sub(chars_to_ignore_regex, "", s)
        s = re.sub(japanese_punct_regex, "", s)
        batch["sentence"] = s
        return batch

    def dataloader_pipeline(self):
        """Load the training and validation datasets."""
        train, valid = self.load_stream_std()
        train, num_samples = self.extract_samples(train)
        valid = self.extract_valid_samples(valid, num_samples)

        # remove punctuation
        train = train.map(self.remove_punctuation)
        valid = valid.map(self.remove_punctuation)

        # Set the sampling rate to 16k
        train = train.cast_column("audio", Audio(sampling_rate=16000))
        valid = valid.cast_column("audio", Audio(sampling_rate=16000))
        return train, valid

    def gen_from_iterable_dataset(self, iterable_dataset: IterableDataset):
        """For converting IterableDataset into a Dataset.
        For example:
        train = Dataset.from_generator(partial(self.gen_from_iterable_dataset, train), features=train.features) 
        """
        yield from iterable_dataset
    
    def load_dataset_stream(self, split: str):
        """Load dataset from Common Voice."""
        dataset = load_dataset(DATASET_REPO,
                               self.lang,
                               split=split,
                               streaming=True)
        return dataset

    def list2dataset(self, dataset: List[Dict]):
        """Convert a list of dicts (batches) into Dataset."""
        return Dataset.from_pandas(pd.DataFrame(data=dataset))

    def load_stream_std(self) -> tuple:
        """Load the train, valid, test sets with the streaming mode.
        If the language is Korean or English, the dataset is loaded from Zeroth-Korean
        or Librispeech ASR (English) downloaded with the non-streaming mode."""
        if self.lang == "ko":
            train, valid = load_dataset(ZEROTH_KOREAN,
                                        split=["train[:80%]", "train[-20%:]"])
            train = train.rename_column("text", "sentence")
            valid = valid.rename_column("text", "sentence")
            return train, valid
        elif self.lang == "en":
            train = load_dataset(LIBRISPEECH,
                                 split="train.clean.100")
            valid = load_dataset(LIBRISPEECH,
                                 split="validation.clean")
            train = train.rename_column("text", "sentence")
            valid = valid.rename_column("text", "sentence")
            return train, valid
        else:
            train = self.load_dataset_stream("train")
            valid = self.load_dataset_stream("validation")
            return train, valid

    def extract_samples(self, dataset: IterableDataset) -> tuple:
        """
	Extract samples until the total audio length reaches the specified value.
        From previous observations, ASR models can be fine-tuned
        with at least 1k-2k samples from common voice, where
        each sample is a few seconds (~6 secs?).
        Given this observation, we use 10,000 secs as the default limit.
        """
        total_sec = 0
        new_dataset = []
        for d in dataset:
            sr = d["audio"]["sampling_rate"]
            arr = d["audio"]["array"]
            sec = len(arr) / sr
            d["audio"]["length"] = sec
            if sec > self.max_audio_len: # Remove long data
                continue
            if self.max_sample > total_sec:
                new_dataset.append(d)
                total_sec += sec
            else: # Reached the limit
                print("Reached the limit.")
                break
        num_samples = len(new_dataset)
        new_dataset = self.list2dataset(new_dataset)
        return new_dataset, num_samples

    def extract_valid_samples(self,
                              dataset: IterableDataset,
                              num_samples: int) -> Dataset:
        """Extract 1/8 of the training data, based on the 8:1:1 rule.

        Arguments:
        - dataset (IterableDataset): a validation dataset.
        - num_samples (int): the number of samples in the validation dataset.

        Return:
        - Dataset: the validation dataset with the require number of samples."""
        num_valid = num_samples // 8
        if type(dataset) == IterableDataset:
            dataset = dataset.take(num_valid)
        elif type(dataset) == Dataset:
            dataset = dataset.select(range(num_valid))
        dataset = self.list2dataset(dataset)
        return dataset

class Converter:
    def __init__(self, lang: str, mode: str, uncased: bool):
        """Converter class.
        Args:
        - lang (str): the language. Specify the ISO code used in the dataset.
        - mode (str): the mode of conversion. kana, romaji, jamo, etc.
        - uncased (bool): 
        """
        assert mode in {"kanji", "kana", "romaji", "jamo", "pinyin", "zhuyin", "latin", None}

        self.lang = lang
        self.mode = mode
        self.uncased = uncased
        if lang == "ja":
            self.kks = pykakasi.kakasi()
            self.tokenizer_obj = sudachipy.dictionary.Dictionary().create()
            self.tokenize_mode = sudachipy.tokenizer.Tokenizer.SplitMode.C # longest segmentation

        # test
        if mode == "jamo":
            assert lang == "ko"
        if mode in {"kanji", "kana", "romaji"}:
            assert lang == "ja"
        if mode in {"pinyin", "zhuyin"}:
            assert lang in {"zh-CN", "zh-TW"}

    def converter_pipeline(self, train, valid):
        """Converter pipeline."""
        train = self.convert(train)
        valid = self.convert(valid)
        return train, valid
        
    def convert(self, dataset: Dataset):
        """Convert sentences into kana or romaji.
        """
        if self.mode == "kanji":
            dataset = dataset.map(self.tokenize_kanjikana)
            return dataset
        elif self.mode == "jamo": # Korean Jamo
            dataset = dataset.map(self.to_jamo)
            return dataset
        elif self.mode == "pinyin": # Chinese Pinyin
            dataset = dataset.map(self.to_pinyin)
            return dataset
        elif  self.mode == "zhuyin": # Chinese Zhuyin
            dataset = dataset.map(self.to_zhuyin)
            return dataset
        elif self.lang == "tt" and self.mode == "latin":
            dataset = dataset.map(self.to_tt_latin)
            return dataset
        elif self.lang == "th" and self.mode == "latin":
            dataset = dataset.map(self.to_th_latin)
            return dataset
        elif self.mode == None:
            # no conversion needed
            if self.uncased:
                dataset = dataset.map(self.lowercase)
            return dataset
        elif self.mode == "kana" or self.mode == "romaji":
            dataset = dataset.map(self.to_kana)
            if self.mode == "kana":
                return dataset
            else:
                dataset = dataset.map(self.to_roma)
                return dataset
        else:
            raise NotImplementedError()

    def lowercase(self, batch: dict) -> dict:
        """.lower() but for Dataset.map()"""
        batch["sentence"] = batch["sentence"].lower()
        return batch

    def tokenize_kanjikana(self, batch: dict) -> dict:
        """Insert whitespaces at word boundaries."""
        text = batch["sentence"]
        words = [m.surface() for m in self.tokenizer_obj.tokenize(text, self.tokenize_mode)]
        batch["sentence"] = " ".join(words)
        return batch

    def to_kana(self, batch: dict) -> dict:
        """Convert kanji to kana using sudachipy (for map function)"""
        text = batch["sentence"]
        reading = " ".join([m.reading_form() for m in self.tokenizer_obj.tokenize(text, self.tokenize_mode)])
        batch["sentence"] = reading
        return batch

    def to_roma(self, batch: dict) -> dict:
        """Convert kana to Hepburn romaji (for map function)"""
        text = batch["sentence"].replace("ッ", "q").split() # sokuon replacement
        romaji = []
        for t in text:
            roma = self.kks.convert(t) # -> dict
            roma = "".join([r["hepburn"] for r in roma])
            romaji.append(roma)
        batch["sentence"] = " ".join(romaji)
        return batch

    def to_jamo(self, batch: dict) -> dict:
        """Convert hangul to jamo."""
        text = batch["sentence"].split() # split by whitespace
        jamo = ""
        for w in text:
            for c in w:
                jamo += hangul.hangul2jamo(c)
            jamo += " " # add back whitespace
        batch["sentence"] = jamo
        return batch

    def to_pinyin(self, batch: dict) -> dict:
        """Convert text in Chinese into Pinyin."""
        text = batch["sentence"]
        pinyin = [dragonmapper.hanzi.to_pinyin(char).lower() for char in text]
        batch["sentence"] = " ".join(pinyin)
        return batch

    def to_zhuyin(self, batch: dict) -> dict:
        """Convert text in Chinese into Zhuyin."""
        text = batch["sentence"]
        zhuyin = [dragonmapper.hanzi.to_zhuyin(char) for char in text]
        batch["sentence"] = " ".join(zhuyin)
        return batch

    def to_tt_latin(self, batch: dict) -> dict:
        """Convert Tatar text in Cyrillic into Latin."""
        text = batch["sentence"]
        trans = translit_tt.translit()
        latin = trans.translit(text)
        batch["sentence"] = latin
        return batch

    def to_th_latin(self, batch: dict) -> dict:
        """Transliterate Thai."""
        text = batch["sentence"]
        latin = transliterate(text, engine="tltk_ipa")
        batch["sentence"] = latin
        return batch

class Vocab:
    def __init__(self, train: Dataset, valid: Dataset):
        self.train = train
        self.valid = valid

    def vocab_pipeline(self):
        """Pipeline for creating the vocab file.
        Args:
        Return:
        - num_chartypes (int): the number of character types in the dataset"""
        vocab_train = self.create_vocab(self.train)
        vocab_valid = self.create_vocab(self.valid)
        vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_valid["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        num_chartypes = len(vocab_dict)

        # Add special characters
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        # Create the vocab file
        if not os.path.exists(args.output):
            # assuming that the model and vocab file fall into the same directory
            os.mkdir(args.output)
        with open(args.vocab, "w") as f:
            json.dump(vocab_dict, f)
            print("Vocabulary created")
        if args.vocab_mode:
            exit()
        return num_chartypes
        
    def extract_chars(self, batch: dict) -> dict:
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab],
                "all_text": [all_text]}

    def create_vocab(self, dataset: Dataset) -> Dataset:
        vocab = dataset.map(
            self.extract_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=dataset.column_names)
        return vocab

def preprocess(dataset: Dataset, pretrained_model, num_proc=24) -> Dataset:
    if args.pretrained_model == "wav2vec2":
        dataset = dataset.map(
            prepare_dataset_for_wav2vec2,
            remove_columns=dataset.column_names,
            num_proc=num_proc
        )
    elif args.pretrained_model == "whisper":
        dataset = dataset.map(
            prepare_dataset_for_whisper,
            remove_columns=dataset.column_names,
            num_proc=num_proc
            )
    else:
        raise NotImplementedError
    return dataset

def prepare_dataset_for_wav2vec2(batch: dict) -> dict:
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

def prepare_dataset_for_whisper(batch: dict) -> dict:
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"],
                                                sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def compute_metrics_for_whisper(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str,
                               references=label_str)
    return {"cer": cer}

def compute_metrics_for_wav2vec2(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
                )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

def get_args():
    parser = ArgumentParser(description="Multilingual ASR comparison.")
    parser.add_argument("-l", "--language", type=str, default="en")
    parser.add_argument("--mode", required=False, default=None,
                        help="Conversion mode; only used for convertable languages.")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--vocab_mode", action="store_true")
    parser.add_argument("--uncased", action="store_true")
    parser.add_argument("--max_sample", type=int, default=10_000)
    parser.add_argument("--pretrained_model", type=str,
                        default="wav2vec2")
    parser.add_argument("--model_size", type=str,
                        default="xlsr53")
    parser.add_argument("--tokenizer_language", type=str, default="English")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--load_preprocessed_data", action="store_true")
    args = parser.parse_args()
    if args.output == None:
        model_path = "/afs/crc.nd.edu/group/nlp/01/ctaguchi/hw3models"
        args.output = os.path.join(model_path, args.language)
    if args.vocab == None:
        args.vocab = os.path.join(args.output, "vocab.json")
    assert args.pretrained_model in {"wav2vec2", "whisper"}
    if args.pretrained_model == "wav2vec2":
        assert args.model_size in {"xlsr53", "300m", "1b", "2b", "bert"}
    elif args.pretrained_model == "whisper":
        assert args.model_size in {"tiny", "base", "small", "medium", "large", "large-v2"}
    return args

if __name__ == "__main__":
    print(sys.version)
    args = get_args()
    if args.no_cache:
        from datasets import disable_caching
        disable_caching()

    data_dir = f"data/{args.language}"
    if args.uncased:
        data_dir += "-uncased"
    if args.mode is not None:
        data_dir += f"-{args.mode}"
    data_dir += f"-{args.max_sample}"
    train_path = data_dir + "/train"
    valid_path = data_dir + "/validation"
        
    # Load dataset
    start = time.time()
    if args.load_preprocessed_data and os.path.exists(data_dir):
        train = load_from_disk(train_path)
        valid = load_from_disk(valid_path)
    else:
        dataloader = DataLoader(args.language, max_sample=args.max_sample)
        train, valid = dataloader.dataloader_pipeline()
        # convert
        converter = Converter(args.language, args.mode, args.uncased)
        train, valid = converter.converter_pipeline(train, valid)
        print("Conversion done.")
        # save
        print("Savind the data...")
        train.save_to_disk(train_path)
        valid.save_to_disk(valid_path)
        
    end = time.time()
    print("Time for loading data:", end - start)

    print("Data sample:")
    print(train[0])

    print("Compute unigram entropy.")
    train_valid = concatenate_datasets([train, valid])
    unient = UnigramEntropy()
    unigram_entropy = unient.compute_unigram_entropy(train_valid)
    print("unigram entropy:", unigram_entropy)

    # Shuffle
    train = train.shuffle(seed=42)
    print("Dataset shuffled")

    print("Creating the vocabulary file and the tokenizer...")
    vocab = Vocab(train, valid)
    num_chartypes = vocab.vocab_pipeline()
    print(f"Vocab created with {num_chartypes} character types.")

    # Pretrained model
    if args.model_size == "xlsr53":
        pretrained_model = "facebook/wav2vec2-large-xlsr-53"
    elif args.model_size == "300m":
        pretrained_model = "facebook/wav2vec2-xls-r-300m"
    elif args.model_size == "1b":
        pretrained_model = "facebook/wav2vec2-xls-r-1b"
    elif args.model_size == "2b":
        pretrained_model = "facebook/wav2vec2-xls-r-2b"
    elif args.model_size == "bert":
        pretrained_model = "facebook/w2v-bert-2.0"
    elif args.model_size == "tiny":
        pretrained_model = "openai/whisper-tiny"
    elif args.model_size == "base":
        pretrained_model = "openai/whisper-base"
    elif args.model_size == "small":
        pretrained_model = "openai/whisper-small"
    elif args.model_size == "medium":
        pretrained_model = "openai/whisper-medium"
    elif args.model_size == "large":
        pretrained_model = "openai/whisper-large"
    elif args.model_size == "large-v2":
        pretrained_model = "openai/whisper-large-v2"
    else:
        raise NotImplementedError

    # Tokenizer
    if args.pretrained_model == "wav2vec2":
        tokenizer = Wav2Vec2CTCTokenizer(args.vocab,
                                         unk_token="[UNK]",
                                         pad_token="[PAD]",
                                         word_delimiter_token=" ")
    elif args.pretrained_model == "whisper":
        tokenizer = WhisperTokenizer.from_pretrained(pretrained_model,
                                                     language=args.tokenizer_language,
                                                     task="transcribe")
    else:
        raise NotImplementedError
    print("Tokenizer created")

    print("Defining the feature extractor...")
    # Feature extractor
    if args.pretrained_model == "wav2vec2":
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16000,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=True)
    elif args.pretrained_model == "whisper":
        feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model)
    else:
        raise NotImplementedError
    print("Feature extractor defined")

    print("Defining the processor...")
    if args.pretrained_model == "wav2vec2":
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer)
    elif args.pretrained_model == "whisper":
        processor = WhisperProcessor.from_pretrained(pretrained_model,
                                                    language=args.tokenizer_language,
                                                    task="transcribe")
    else:
        raise NotImplementedError
    print("Processor defined")

    print("Preprocessing the data...")
    # Preprocess the dataset
    train = preprocess(train, args.pretrained_model)
    valid = preprocess(valid, args.pretrained_model)
    print("Preprocessing done")
    
    # data collator
    if args.pretrained_model == "wav2vec2":
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    elif args.pretrained_model == "whisper":
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    else:
        raise NotImplementedError

    # Evaluation metrics
    metric = evaluate.load("cer")

    # Model
    if args.pretrained_model == "wav2vec2":
        model = Wav2Vec2ForCTC.from_pretrained(
            pretrained_model,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer)
        )
        model.freeze_feature_extractor()
    elif args.pretrained_model == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(PRETRAINED)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
    else:
        raise NotImplementedError

    # Output
    if args.pretrained_model == "wav2vec2":
        training_args = TrainingArguments(
            output_dir=args.output,
            group_by_length=True,
            per_device_train_batch_size=args.batch_size,
            # gradient_accumulation_steps=(16 // args.batch_size),
            evaluation_strategy="steps",
            num_train_epochs=args.epoch,
            fp16=args.fp16,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=args.learning_rate,
            warmup_steps=20,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            report_to=None,
            push_to_hub=False,
        )
    elif args.pretrained_model == "whisper":
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTDIR,
            group_by_length=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=(16 // args.batch_size), # change this in wav2vec2 code. increase by 2x for every 2x decrease in batch size.
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            num_train_epochs=args.epoch,
            fp16=args.fp16,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            logging_steps=10,
            learning_rate=args.learning_rate,
            warmup_steps=1000,
            max_steps=-1,
            predict_with_generate=True,
            generation_max_length=225,
            # gradient_checkpointing=True, # <- this can cause segmentation fault; avoid this
            report_to="wandb",
            run_name=args.wandb_run_name,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            push_to_hub=False,
        )
    else:
        raise NotImplementedError

    if args.pretrained_model == "wav2vec2":
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train,
            eval_dataset=valid,
            compute_metrics=compute_metrics_for_wav2vec2,
            tokenizer=processor.feature_extractor,
        )
    elif args.pretrained_model == "whisper":
        trainer = Seq2SeqTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train,
            eval_dataset=valid,
            compute_metrics=compute_metrics_for_whisper,
            tokenizer=processor.feature_extractor
        )
    else:
        raise NotImplementedError
    
    trainer.train()
    # trainer.evaluate()
    # trainer.save_state()
    # trainer.save_model()
