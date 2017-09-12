import sys
import numpy as np

# adds tokens from an array to the unigram frequency table
def add_to_uni_table(token_arr):
    for t in token_arr:
        if t in uni_frequency_table:
            uni_frequency_table[t] = uni_frequency_table[t] + 1
        else:
            uni_frequency_table[t] = 1

# adds tokens from an array to the bigram frequency table, prepending "phi" to beginning of sentence
def add_to_bi_table(token_arr):
    token_arr.insert(0, "phi") # prepend "phi" to beginning of sentence
    for i in range(len(token_arr) - 1):
        bigram = token_arr[i] + " " + token_arr[i+1]
        if bigram in bi_frequency_table:
            bi_frequency_table[bigram] = bi_frequency_table[bigram] + 1
        else:
            bi_frequency_table[bigram] = 1

# Returns the conditional probability (w_i | w_iMinus1)
def conditional_prob_unsmoothed(w_i, w_iMinus1):
    bigram = w_iMinus1 + " " + w_i
    if(bigram not in bi_frequency_table):
        return 0
    prob_wi_AND_wiMinus1 = bi_frequency_table[bigram] / bi_frequencies_sum
    freq_w_iMinus1 = 0 # initialize
    for k,v in bi_frequency_table.items(): # count how many times w_iMinus1 is first part of a bigram
        if k.split()[0] == w_iMinus1:
            freq_w_iMinus1 += v
    prob_wiMinus1 = freq_w_iMinus1 / bi_frequencies_sum
    return (prob_wi_AND_wiMinus1 / prob_wiMinus1)

def conditional_prob_smoothed(w_i, w_iMinus1):
    bigram = w_iMinus1 + " " + w_i
    if(bigram not in bi_frequency_table):
        return 0
    prob_wi_AND_wiMinus1 = bi_frequency_table[bigram] / bi_frequencies_sum
    freq_w_iMinus1 = 0 # initialize
    for k,v in bi_frequency_table.items(): # count how many times w_iMinus1 is first part of a bigram
        if k.split()[0] == w_iMinus1:
            freq_w_iMinus1 += v
    prob_wiMinus1 = freq_w_iMinus1 / bi_frequencies_sum
    return (prob_wi_AND_wiMinus1 / prob_wiMinus1)
    
# Calculates and prints the log-probability of a given sentence using the unigram language model
def print_sentence_prob_unigram(sentence, uni_frequencies_sum):
    token_arr = sentence.lower().split()
    logprob_sum = 0
    for t in token_arr:
        token_prob =  np.log2(uni_frequency_table[t] / uni_frequencies_sum)
        logprob_sum += token_prob

    print("Unsmoothed Unigrams, logprob(S) = " + "{0:.4f}".format(logprob_sum))

# Calculates and prints the log-probability of a given sentence using the bigram language model
def print_sentence_prob_bigram_unsmoothed(sentence, bi_frequencies_sum):
    token_arr = sentence.lower().split()
    token_arr.insert(0, "phi") # prepend "phi" to beginning of sentence
    logprob_sum = 0

    for i in range(1, len(token_arr)):
        w_i = token_arr[i]
        w_iMinus1 = token_arr[i - 1]
        prob = conditional_prob_unsmoothed(w_i, w_iMinus1)
        if prob == 0:
            print("Unsmoothed Bigrams, logprob(S) = undefined")
            return
        logprob = np.log2(prob)
        logprob_sum += logprob

    print("Unsmoothed Bigrams, logprob(S) = " + "{0:.4f}".format(logprob_sum))

# Calculates and prints the log-probability of a given sentence using the bigram language model
# Applies an add-one smoothing
def print_sentence_prob_bigram_smoothed(sentence, bi_frequencies_sum):
    token_arr = sentence.lower().split()
    token_arr.insert(0, "phi") # prepend "phi" to beginning of sentence
    logprob_sum = 0

    for i in range(1, len(token_arr)):
        w_i = token_arr[i]
        w_iMinus1 = token_arr[i - 1]
        prob = conditional_prob_smoothed(w_i, w_iMinus1)
        logprob = np.log2(prob)
        logprob_sum += logprob

    print("Smoothed Bigrams, logprob(S) = " + "{0:.4f}".format(logprob_sum))

### START OF PROGRAM ###
if(len(sys.argv) < 4):
    print("This program requires 3 commandline arguments: <training-file> -test <test-file> \n Please try again")
    sys.exit()

uni_frequency_table = {}
bi_frequency_table = {}
training_file = open(sys.argv[1])

for sentence in training_file:
    tokens = sentence.lower().split()  # makes tables case-insensitive
    add_to_uni_table(tokens)
    add_to_bi_table(tokens)

# Count the frequencies of all unigrams in the table
uni_frequencies_sum = 0
for k,v in uni_frequency_table.items():
    uni_frequencies_sum += v

# Count the frequencies of all bigrams in the table
bi_frequencies_sum = 0
for k,v in bi_frequency_table.items():
    bi_frequencies_sum += v

# Calculate and print sentence probabilities from the given test file
test_file = open(sys.argv[3])
for sentence in test_file:
    print("S = ",sentence)
    print_sentence_prob_unigram(sentence, uni_frequencies_sum)
    print_sentence_prob_bigram_unsmoothed(sentence, bi_frequencies_sum)
    #print_sentence_prob_bigram_smoothed(sentence, bi_frequencies_sum)
    print()
