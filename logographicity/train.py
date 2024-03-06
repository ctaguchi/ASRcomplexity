import eng_to_ipa
import numpy as np
from typing import Any, Tuple, List
import tensorflow as tf
import tensorflow_text as tf_text
import einops
import random
import re
from argparse import ArgumentParser
import string
import pickle
import os
import gruut
import multiprocessing

# Local
import logographicity_metrics as metrics

WINDOW = 3 # context window

def get_args():
    """Argument Parser."""
    parser = ArgumentParser()
    parser.add_argument("-l", "--language", type=str, default="en",
                        help="The language of the data.")
    parser.add_argument("-d", "--data", type=str,
                        default="./data/en/europarl-v7.bg-en.en",
                        help="The path to the training data.")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="The path to the evaluation data.")
    parser.add_argument("-s", "--samples", type=int, default=1000,
                        help="The number of samples for training the model.")
    parser.add_argument("--eval_samples", type=int, default=10000,
                        help="The number of samples for evaluating the model.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="The number of epochs for training the model.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="The path to the output stats file.")
    parser.add_argument("--force_dataload", action="store_true")
    parser.add_argument("--reload_model", action="store_true",
                        help="""Reload the model instead of training from scratch.
                        This is not recommended because it cannot retrieve the
                        last attention weights.""")
    parser.add_argument("--num_proc", default=multiprocessing.cpu_count(),
                        help="Number of processors to use for multiprocessing.")
    parser.add_argument("--read_data_only", action="store_true",
                        help="Reading data only, and don't train the model.")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.language}_{args.samples}_stats.csv"
    return args

class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return
        
        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)
      
            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")

def g2p(l: str):
    """Grapheme to phoneme conversion for preprocessing.

    args:
    - l (str): line (sentence).
    """
    if args.language == "en":
        ipa = eng_to_ipa.convert(l)
        ipa = ipa.replace("ˌ", "").replace("ˈ", "")
        ipa_chars, target_word = mask_context(l, ipa)
        assert "<TARG>" in ipa_chars.split(), print(ipa_chars)
    elif args.language in {"sv", "fr", "cs", "it", "nl", "de", "es", "ru", "sw"}: # covered in gruut
        sents = gruut.sentences(l, args.language)
        ipa_l = [(" ".join(word.phonemes), word.text) for sent in sents for word in sent if word.phonemes is not None]
        ipa, l = zip(*ipa_l)
        ipa_chars, target_word = mask_context_for_gruut(list(l), list(ipa))
    return ipa_chars, target_word

def remove_stress(s: str) -> str:
    s = s.translate(str.maketrans("", "", "ˈˌ"))
    return s

def remove_stress_from_data(data: list) -> list:
    return [(remove_stress(d[0]), d[1]) for d in data]

def g2p_multitargets(l: str) -> list:
    """Grapheme to phoneme conversion for preprocessing,
    with multiple targets per sentence for data-augmentation.

    args:
    - l (str): a line (sentence).
    return:
    - samples (list): a list of samples processed from a sentence.
    """
    samples = []
    if args.language == "en":
        ipa = eng_to_ipa.convert(l)
        ipa = remove_stress(l)        
        # multiple targets
        ipa_tokens = ipa.split()
        for i in range(WINDOW, len(ipa_tokens)-WINDOW): # 3 for window size
            ipa_chars, target_word = mask_context(l, ipa, i)
            samples.append((ipa_chars, target_word))
        # assert "<TARG>" in ipa_chars.split(), print(ipa_chars)
    elif args.language in {"sv", "fr", "cs", "it", "nl", "de", "es", "ru", "sw"}: # covered in gruut
        sents = gruut.sentences(l, args.language)
        ipa_word = [(" ".join(word.phonemes), word.text) for sent in sents for word in sent if word.phonemes is not None]
        # print(ipa_word)
        try:
            ipas, words = zip(*ipa_word) # -> Tuple[str], Tuple[str]; ('h ɛ l ˈoʊ', 'w ˈɚ l d'), ('hello', 'world')
            ipas = [remove_stress(ipa) for ipa in ipas]
        except:
            print("No IPA/words detected.", ipa_word)
            return
        # multiple targets
        # ipa_tokens = ipa.split()
        for i in range(WINDOW, len(ipas)-WINDOW):
            ipa_chars, target_word = mask_context_for_gruut(list(words), ipas, i)
            samples.append((ipa_chars, target_word))
    return samples

def read_data(path: str) -> list:
    """Just read the data."""
    with open(path, "r") as f:
        lines = f.readlines()
    # remove samples with less than 7 words (for context window of 3)
    # remove samples with numbers
    # rstrip() to remove "\n"
    # lowercase the sentence and remove punctuation symbols
    lines = [l.rstrip() for l in lines]
    lines = [l for l in lines if not any(c.isdigit() for c in l)]
    lines = [l.lower().translate(str.maketrans("", "", string.punctuation)) for l in lines]
    lines = [l for l in lines if len(l.split()) >= WINDOW*2+1]
    return lines

def mask_context(eng: str, ipa: str, mask_idx=0) -> Tuple[str, str]:
    """Randomly select a target word from a sentence.
    arg:
    - eng (str): an English sentence.
    - ipa (str): IPA transcription of eng
    - idx (int): Index of the token to be masked. If 0 (default), randomly assign an index.
    return:
    - ipa (str): IPA transcription with <TARG> tags
    - target_word (str): Targeted word in the language; split by whitespace
    """
    eng_tokens = eng.split()
    # mask_idx = random.randrange(len(eng_tokens))
    if mask_idx == 0:
        # mask_idx = random.randrange(1, min(MASK_TOKEN_RANGE, len(eng_tokens)) - 1)
        mask_idx = random.randrange(WINDOW, len(eng_tokens)-WINDOW)
    target_word = eng_tokens[mask_idx]
    target_chars = " ".join(list(target_word))
    
    ipa_tokens = ipa.split()
    ipa_tokens.insert(mask_idx, "<TARG>")
    ipa_tokens.insert(mask_idx+2, "</TARG>")
    ipa_chars = ""
    for i, token in enumerate(ipa_tokens):
        if token not in {"<TARG>", "</TARG>"}:
            char_spaced = " ".join(list(token))
            if i == len(ipa_tokens) - 1:
                ipa_chars += char_spaced
            elif i == mask_idx + 1: # the target token
                ipa_chars += char_spaced
            elif i ==  mask_idx - 1: # before the target token
                ipa_chars += char_spaced
            else:
                ipa_chars += (char_spaced + " [SEP] ")
        else:
            if i == len(ipa_tokens) - 1:
                ipa_chars += (" " + token)
            elif i == 0:
                ipa_chars += (token + " ")
            else:
                ipa_chars += (" " + token + " ")        
    return ipa_chars, target_chars

def separator_for_gruut(list_ipa: list) -> str:
    """Insert separator tags [SEP]."""
    ipa_tokens_sep = []
    tags = {"<TARG>", "</TARG>", "[SEP]"}
    for i, ipa in enumerate(list_ipa):
        ipa_tokens_sep.append(ipa)
        if i != len(list_ipa)-1:
            if list_ipa[i+1] not in tags and list_ipa[i] not in tags:
                ipa_tokens_sep.append("[SEP]")
    # ipa_tokens_sep -> ["'h ɛ l ˈoʊ", "[SEP]", "w ˈɚ l d", "[SEP]", "ð ˈɪ s", "<TARG> "ˈɪ z", "</TARG>", "p ˈaɪ θ ɑ n", "[SEP]", 'j ˈu', "[SEP]", 'ˈɑ ɹ', "[SEP]", 'p ɹ ˈoʊ ɡ ɹ ˌæ m ɚ']
    ipa_chars = " ".join(ipa_tokens_sep)
    return ipa_chars

def mask_context_for_gruut(sent: list, ipa: list, mask_idx=0):
    """Mask the targeted word from the context sentence.
    Originally designed for sentences preprocessed with gruut,
    but this function is probably going to be the mainstream one.

    args:
    - sent (list): The original sentence; list of tokens; List[str]; ['hello', 'world']
    - ipa (list): List of corresponding IPA transcriptions; List[str]; ['h ɛ l ˈoʊ', 'w ˈɚ l d']
    - mask_idx (int): The index in the token list in which the token is masked
    - window (int): The window size of context tokens around the target.
    Following Sproat & Gutkin (2021), the window size is set to 3 by default.
    """
    if mask_idx == 0:
        mask_idx = random.randrange(WINDOW, len(sent)-WINDOW)
    target_word = sent[mask_idx]
    target_chars = " ".join(list(target_word))
    # non-destructive way
    assert len(ipa) >= WINDOW*2 + 1, (len(ipa), ipa) # context length must be larger than 7, so that the target can take a window of 3 tokens on both sides
    left_ctx = ipa[mask_idx-WINDOW:mask_idx]
    right_ctx = ipa[mask_idx+1:mask_idx+WINDOW+1]
    ipa_with_trg = left_ctx + ["<TARG>"] + [ipa[mask_idx]] + ["</TARG>"] + right_ctx
    ipa_chars = separator_for_gruut(ipa_with_trg)
    return ipa_chars, target_chars

def tf_normalize(text):
    text = tf.strings.lower(text) # -> [SEP] will become [sep], <TARG> will become <targ>, but it's okay
    # remove punctuation
    text = tf.strings.regex_replace(text, "[.,?!:;\"\']", "")

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def process_text(context, target):
    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)
    targ_in = target[:,:-1].to_tensor()
    targ_out = target[:,1:].to_tensor()
    return (context, targ_in), targ_out

# Model
UNITS = 256
class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units
    
        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                                   mask_zero=True)

        # The RNN layer processes those vectors sequentially.
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform'))

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        # 2. The embedding layer looks up the embedding vector for each token.
        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        # 3. The GRU processes the sequence of embeddings.
        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        # 4. Returns the new sequence of embeddings.
        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        shape_checker = ShapeChecker()
    
        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)
        
        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')
        
        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units


        # 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                units, mask_zero=True)

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = tf.keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self,
             context, x,
             state=None,
             return_state=False):  
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch t')
        shape_checker(context, 'batch s units')

        # 1. Lookup the embeddings
        x = self.embedding(x)
        shape_checker(x, 'batch t units')

        # 2. Process the target sequence.
        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, 'batch t units')

        # 3. Use the RNN output as the query for the attention over the context.
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, 'batch t units')
        shape_checker(self.last_attention_weights, 'batch t s')

        # Step 4. Generate logit predictions for the next token.
        logits = self.output_layer(x)
        shape_checker(logits, 'batch t target_vocab_size')

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result

    def get_next_token(self, context, next_token, done, state, temperature = 0.0):
        logits, state = self(
            context, next_token,
            state = state,
            return_state=True) 
    
        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :]/temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        # If a sequence produces an `end_token`, set it `done`
        done = done | (next_token == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
    
        return next_token, done, state

class Translator(tf.keras.Model):
    def __init__(self, units,
                context_text_processor,
                target_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        #TODO(b/250038731): remove this
        try:
            # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def translate(self,
                  texts, *,
                  max_length=50,
                  temperature=0.0):
        # Process the input texts
        context = self.encoder.convert_input(texts)
        batch_size = tf.shape(texts)[0]

        # Setup the loop inputs
        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        for _ in range(max_length):
            # Generate the next token
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done,  state, temperature)
            
            # Collect the generated tokens
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)
        
            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Stack the lists of tokens and attention weights.
        tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

        result = self.decoder.tokens_to_text(tokens)
        return result

    def get_last_attention_weights(self):
        """for getting last attention weights"""
        return self.last_attention_weights[0]

def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)

if __name__ == "__main__":
    args = get_args()

    data_pickled = args.data + str(args.samples)
    eval_data_pickled = args.data + str(args.eval_samples) + "eval"
    if os.path.exists(data_pickled) and not args.force_dataload:
        print("Loading the pre-saved data.")
        with open(data_pickled, "rb") as f:
            data = pickle.load(f)
    else:
        if args.language == "ja":
            raise NotImplementedError
        data_raw = read_data(args.data)
        p = multiprocessing.Pool(args.num_proc)
        data = []
        eval_data = []
        count = 0
        load_eval = False
        for samples in p.map(g2p_multitargets, data_raw):
            # remove fallback ipa transcriptions
            samples = [s for s in samples if "*" not in s]
            print(samples)
            if not load_eval:
                if count <= args.samples:
                    data += samples
                    count += len(samples)
                else:
                    with open(data_pickled, "wb") as f:
                        pickle.dump(data, f)
                    load_eval = True
            else:    
                if args.samples < count <= args.samples + args.eval_samples:
                    eval_data += samples
                    count += len(samples)
                else:
                    with open(eval_data_pickled, "wb") as f:
                        pickle.dump(eval_data, f)
                    break
    if args.read_data_only:
        exit()

    if os.path.exists(f"models/{args.language}_{args.samples}") and args.reload_model:
        model = tf.saved_model.load(f"models/{args.language}_{args.samples}")
        inputs = [context for (context, _) in data[:3]]
        _ = model.translate(tf.constant(inputs)) # warmup
    else:
        context, target = zip(*data)
        context_raw = np.array(context)
        target_raw = np.array(target)

        BUFFER_SIZE = len(context_raw)
        BATCH_SIZE = 64

        is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

        train_raw = (
            tf.data.Dataset
            .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE))
        val_raw = (
            tf.data.Dataset
            .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE))

        max_vocab_size = 5000
        context_text_processor = tf.keras.layers.TextVectorization(
            standardize=tf_normalize,
            max_tokens=max_vocab_size,
            ragged=True)
        context_text_processor.adapt(train_raw.map(lambda context, target: context))

        target_text_processor = tf.keras.layers.TextVectorization(
            standardize=tf_normalize,
            max_tokens=max_vocab_size,
            ragged=True)
        target_text_processor.adapt(train_raw.map(lambda context, target: target))

        train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
        val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

        model = Translator(UNITS, context_text_processor, target_text_processor)

        model.compile(optimizer='adam',
                      loss=masked_loss, 
                      metrics=[masked_acc, masked_loss])
        
        # train
        history = model.fit(
            train_ds.repeat(), 
            epochs=args.epochs, # 5 in the original paper
            steps_per_epoch=len(context_raw)*0.8//BATCH_SIZE,
            validation_data=val_ds,
            validation_steps=len(context_raw)*0.2//BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3)])

        # Save the model
        export = Export(model)
        # To compile, we need to run tf.function once
        inputs = [sent.decode() for cons, tgts in train_raw.take(1) for sent in cons.numpy()[:3]]
        _ = export.translate(tf.constant(inputs))
        tf.saved_model.save(export, f"models/{args.language}_{args.samples}",
                        signatures={'serving_default': export.translate})

    # Evaluation
    s_words = 0
    sw_greedy = 0
    sw_prob = 0
    sw_sigmoid = 0

    import tqdm
    if args.eval_data is None:
        try:
            eval_data
        except NameError:
            eval_data = data
    else:
        with open(args.eval_data, "rb") as f:
            eval_data = pickle.load(f)
    EVAL_SAMPLES = 10000
    num_correct = 0
    num_non_one_tokens = 0 # exclude target tokens with only one character
    for ipa, target_chars in tqdm.tqdm(eval_data[:EVAL_SAMPLES]):
        sent_attn, masked_sent_attn, pred = metrics.inference_and_get_attn(model, ipa)
        print("Prediction:", pred)
        pred_concat = pred.lower().split()
        print("Pred_concat:", pred_concat)
        target_chars_concat = target_chars.lower().split()
        print("Reference:", target_chars)
        print("Target_concat:", target_chars_concat)
        if pred_concat != target_chars_concat:
            # Don't count wrongly predicted words
            print("Prediction is wrong.")
            continue

        print("Prediction is correct.")
        s_word = metrics.attn_spread(sent_attn, masked_sent_attn)
        print("S_word:", s_word)
        s_words += s_word
        num_correct += 1

        if len(target_chars_concat) == 1:
            continue
        else:
            num_non_one_tokens += 1
            
        wordattnspread = metrics.WordAttnSpread(ipa, sent_attn.numpy())
        word_attn = wordattnspread.attn # -> ndarray

        masked_attn_greedy = wordattnspread.greedy_mask()
        masked_attn_prob = wordattnspread.prob_mask()
        masked_attn_sigmoid = wordattnspread.sigmoid_mask()
        assert not np.isnan(masked_attn_greedy).any(), print(ipa, word_attn)
        assert not np.isnan(masked_attn_prob).any(), print(ipa, word_attn)
        assert not np.isnan(masked_attn_sigmoid).any(), print(ipa, word_attn)

        sw_greedy_word = wordattnspread.word_sw_score(masked_attn_greedy)
        print("Greedy Sw:", sw_greedy_word)
        sw_prob_word = wordattnspread.word_sw_score(masked_attn_prob)
        print("Probabilistic Sw:", sw_prob_word)
        sw_sigmoid_word = wordattnspread.word_sw_score(masked_attn_sigmoid)
        print("Sigmoid Sw:", sw_sigmoid_word)
        sw_greedy += sw_greedy_word
        sw_prob += sw_prob_word
        sw_sigmoid += sw_sigmoid_word
        assert not np.isnan(sw_greedy).any(), print(ipa, word_attn)
        assert not np.isnan(sw_prob).any(), print(ipa, word_attn)
        assert not np.isnan(sw_sigmoid).any(), print(ipa, word_attn)

    s_token = metrics.corpus_attn_spread(s_words, EVAL_SAMPLES)
    st_greedy = metrics.corpus_attn_spread(sw_greedy,num_non_one_tokens)
    st_prob = metrics.corpus_attn_spread(sw_prob, num_non_one_tokens)
    st_sigmoid = metrics.corpus_attn_spread(sw_sigmoid, num_non_one_tokens)

    with open(args.output, "w") as f:
        f.write(f"Language,{args.language}\n")
        f.write(f"Train_samples,{args.samples}\n")
        f.write(f"Epochs,{args.epochs}\n")
        f.write(f"Eval_samples,{EVAL_SAMPLES}\n")
        f.write(f"Num_non_one_tokens,{num_non_one_tokens}\n")
        f.write(f"Num_correct,{num_correct}\n")
        f.write(f"S_token,{s_token}\n")
        f.write(f"St_greedy,{st_greedy}\n")
        f.write(f"St_prob,{st_prob}\n")
        f.write(f"St_sigmoid,{st_sigmoid}\n")
