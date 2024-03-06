import numpy as np
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser
import eng_to_ipa
import string
import os

# Local
import train

def get_args():
    """Argument Parser."""
    parser = ArgumentParser("Functions for logographicity metirics.")
    parser.add_argument("-m", "--model",
                        help="The path to the model to be loaded")
    parser.add_argument("-d", "--data",
                        help="The path to the data for evaluation")
    parser.add_argument("-l", "--language",
                        help="The language of the data. (ISO code)")
    parser.add_argument("-s", "--samples", type=int, default=1000,
                        help="The number of samples.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="The path to the output stats file.")
    args = parser.parse_args()
    if args.output is None:
        output_dir = os.path.split(args.model)[0] # directory name
        args.output = os.path.join(output_dir, "stats.csv")
    return args

def load_eval_data(path: str, language, num_samples=1000):
    """Load the data for evaluation."""
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [l.translate(str.maketrans('', '', string.punctuation)).rstrip() for l in lines]
    data = []
    count = 0
    for l in lines:
        if not any(c.isdigit() for c in l):
            # exclude data with numbers
            if language == "en":
                ipa = eng_to_ipa.convert(l)
            else:
                raise NotImplementedError
            if "*" not in ipa:
                # If it's correctly transcribed into IPA, go on
                # Remove accents
                ipa = ipa.replace("ˌ", "").replace("ˈ", "")
                ipa_chars, target_word = train.mask_context(l, ipa)
                data.append((ipa_chars, target_word))
                count += 1
                if count == num_samples:
                    break
    return data

def masked_attn_matrix(inp: str,
                       attn: tf.Tensor):
    """Generate a masked attention matrix."""
    print(inp)
    trg_start_idx = inp.split().index("<TARG>")
    trg_end_idx = inp.split().index("</TARG>")
    shape = attn[:-1, trg_start_idx+2:trg_end_idx+1].shape
    # ^ trg_start_idx+2, because [START] tag will be inserted in the output

    arr = attn.numpy()
    arr[:-1, trg_start_idx+2:trg_end_idx+1] = np.zeros(shape)
    masked_attn = tf.constant(arr)
    return masked_attn

def inference_and_get_attn(model, ipa: str):
    """
    args:
    - ipa: IPA input (context)
    return:
    - attn (tf.Tensor): Attention matrix.
    - masked_attn (tf.Tensor): Masked attention matrix.
    """
    result = model.translate([ipa])
    attn = model.get_last_attention_weights()
    print("inference_and_get_attn: attn", attn.shape, attn)
    if len(attn.shape) == 3:
        attn = attn[0] # old models may have three dimensions in the original attn matrix.
        # fixed in the latest version of the `Translator` class.
        print(attn)
    pred = result[0].numpy().decode()

    # masked attention matrix (masked word)
    masked_attn = masked_attn_matrix(ipa, attn)
    return attn, masked_attn, pred

def attn_spread(attn: tf.Tensor, masked_attn: tf.Tensor):
    """Calculate attention spread per word.
    args:
    - attn: Original `last_attention_weights`
    - masked_attn: masked attention matrix
    return:
    - s_word (float): word-level attention spread
    """
    denom = tf.reduce_sum(attn).numpy()
    numer = tf.reduce_sum(masked_attn).numpy()
    s_word = numer / denom
    return s_word

def corpus_attn_spread(s_words: float, corpus_size: int):
    """Calculate the average attention spread over the corpus.
    args:
    - s_words (float): The sum of every s_word
    - corpus_size (int): The size of the corpus. By default, the number of samples
    return:
    - s_token (float)
    """
    s_token = s_words / corpus_size
    return s_token    

class WordAttnSpread:
    def __init__(self, ipa: str, attn: np.ndarray):
        """
        Calculate the attention spread inside the target word.
        args:
        - attn: original attention matrix in a sentence.
        """
        self.ipa = ipa
        print(attn.shape)
        print(attn)

        self.attn = self.get_word_attn(attn) # word_attn
        assert len(self.attn.shape) == 2, print(self.attn.shape)
        print(self.attn)
        self.row = self.attn.shape[0]
        self.col = self.attn.shape[1]
        self.incl = self.row / self.col

    def get_word_attn(self, attn: np.ndarray, normalize=False):
        """Extract the attention matrix of the target word.
        The extracted matrix can be optionally normalized with softmax (each row sums up to 1).
        args:
        - attn: original attention matrix in a sentence.
        return:
        - word_attn: word attention matrix.
        """
        def np_softmax(array):
            """Softmax function with numpy."""
            mx = np.max(array, axis=-1, keepdims=True)
            numerator = np.exp(array - mx)
            denominator = np.sum(numerator, axis=-1, keepdims=True)
            return numerator / denominator
        
        trg_start_idx = self.ipa.split().index("<TARG>")
        trg_end_idx = self.ipa.split().index("</TARG>")
        # if normalize:
            # word_attn = np_softmax(attn[1:-1, trg_start_idx:trg_end_idx])
        #else:
        # word_attn = attn[1:-1, trg_start_idx+1:trg_end_idx+1]
        word_attn = attn[:-1, trg_start_idx+2:trg_end_idx+1] # < changed feb 2 2024
        # ^ trg_start_idx+2, because the model will insert [START] in the beginnning of the output
        return word_attn

    def greedy_mask(self):
        """Greedy masking.
        Zero out all the elements that the diagonal line crosses.
        """
        masked_attn = self.attn.copy()
        for i in range(self.row):
            for j in range(self.col):
                if i < self.incl * (j+1) <= i+1:
                    masked_attn[i,j] = 0
                elif i <= self.incl * j < i+1:
                    masked_attn[i,j] = 0
        return masked_attn

    def prob_mask(self):
        """Probabilistic masking."""
        mask_matrix = np.ones(self.attn.shape)
        for i in range(self.row):
            for j in range(self.col):
                if i <= self.incl * j <= i+1 or i < self.incl * (j+1) <= i+1:
                    # at least left or right edge crosses
                    if not i < self.incl * (j+1) <= i+1:
                        # right edge does not cross
                        # compute the area of the triangle
                        vert = (i+1) - self.incl * j
                        hori = vert / self.incl
                        mask_matrix[i,j] = 1 - vert * hori # / 2
                        # ^ subtract from 1, since we want to decrease the masking
                        # where the crossing area is small
                    elif self.incl * j < i:
                        # left edge does not cross
                        vert = self.incl * (j + 1) - i
                        hori = (j + 1) - i / self.incl
                        mask_matrix[i,j] = 1 - vert * hori # / 2
                    else:
                        # both edges cross
                        # compute the area of the trapezoid; height is always 1
                        left = self.incl * j - i
                        right = self.incl * (j+1) - i
                        mask_matrix[i,j] = 1 - min(1, (left + right)) # / 2)
        masked_attn = self.attn * mask_matrix # Hadamard product
        return masked_attn

    def sigmoid_mask(self):
        """Sigmoid masking.
        The sigmoid function has the domain and range [0,1].
        """
        def zero_one_sigmoid(x, beta=3):
            """Sigmoid function with the domain and range [0,1].
            Source: https://stats.stackexchange.com/questions/214877/is-there-a-formula-for-an-s-shaped-curve-with-domain-and-range-0-1
            """
            # y = 1 / (1 + (x / (1-x))**(-beta))
            # f(x) = x^b/(x^b + (1-x)^b)
            y = x ** beta / (x**beta + (1-x)**beta)
            return y

        mask_matrix = np.ones(self.attn.shape)
        for i in range(self.row):
            for j in range(self.col):
                if i <= self.incl * j <= i+1 or i < self.incl * (j+1) <= i+1:
                    # at least left or right edge crosses
                    if self.incl * (j+1) > i+1:
                        # right edge does not cross
                        # compute the area of the triangle
                        vert = (i+1) - self.incl * j
                        hori = vert / self.incl
                        prob = 1 - vert * hori
                        # ^ subtract from 1, since we want to decrease the masking
                        # where the crossing area is small
                    elif self.incl * j < i:
                        # left edge does not cross 
                        vert = self.incl * (j + 1) - i
                        hori = (j + 1) - i / self.incl
                        prob = 1 - vert * hori
                    else:
                        # both edges cross
                        # compute the area of the trapezoid; height is always 1
                        left = self.incl * j - i
                        right = self.incl * (j+1) - i
                        prob = 1 - min(1, (left + right))
                    try:
                        mask_matrix[i,j] = zero_one_sigmoid(prob)
                    except ZeroDivisionError:
                        mask_matrix[i,j] = 1
        masked_attn = self.attn * mask_matrix # Hadamard product
        return masked_attn

    def word_sw_score(self, masked_attn: np.ndarray):
        """Calculate the word-level logography score per word
        In other words, word-internal attention spread"""
        denom = np.sum(self.attn)
        numer = np.sum(masked_attn)
        sw_word = numer / denom
        return sw_word

if __name__ == "__main__":
    args = get_args()

    # load data
    print("Loading data...")
    data = load_eval_data(args.data, args.language)

    print("Loading the model...")
    model = tf.saved_model.load(args.model)

    # warmup
    inputs = [d[0] for d in data[:3]]
    _ = model.translate(tf.constant(inputs))
    print(model.decoder.attention.last_attention_weights[0])

    # calculate logographicity scores
    print("Calculating the scores...")
    s_words = 0
    sw_greedy = 0 # new method; total scores with greedy masking
    sw_prob = 0 # new method; total scores with probabilistic masking
    sw_sigmoid = 0 # new method; total scores with sigmoid masking
    
    for ipa, _ in data:
        attn, masked_attn = inference_and_get_attn(model, ipa)
        s_words += attn_spread(attn, masked_attn)
        
        wordattnspread = WordAttnSpread(ipa, attn.numpy())
        
        masked_attn_greedy = wordattnspread.greedy_mask()
        masked_attn_prob = wordattnspread.prob_mask()
        masked_attn_sigmoid = wordattnspread.sigmoid_mask()

        sw_greedy += wordattnspread.word_sw_score(masked_attn_greedy)
        sw_prob += wordattnspread.word_sw_score(masked_attn_prob)
        sw_sigmoid += wordattnspread.word_sw_score(masked_attn_sigmoid)
        
    s_token = corpus_attn_spread(s_words, args.samples)
    st_greedy = corpus_attn_spread(sw_greedy, args.samples)
    st_prob = corpus_attn_spread(sw_prob, args.samples)
    st_sigmoid = corpus_attn_spread(sw_sigmoid, args.samples)

    # file write
    with open(args.output, "w") as f:
        f.write(f"Language,{args.language}\n")
        f.write(f"S_token,{s_token}\n")
        f.write(f"St_greedy,{st_greedy}\n")
        f.write(f"St_prob,{st_prob}\n")
        f.write(f"St_sigmoid,{st_sigmoid}")
