import numpy as np
import pandas as pd
import re

def get_data(data_file):
    # read data file into a pandas dataframe
    with open(data_file, encoding='utf8', errors='ignore') as data:
        essay_df = pd.read_table(data, sep='\t')
    return essay_df

def vectorize_essays(essays, embed_size, verbose=False):
    '''
    * NOTE * This function will only run if the word embedding text file is
    downloaded on the user's local machine. The relevant word embeddings may
    be found here: https://nlp.stanford.edu/projects/glove/
    They are also discussed in more detail in the research paper.
    
    This function will convert a list of essays as strings into a list of 
    essays in word embedded form.
    :param essays: An array of essays; each essay is in the form of a string
    :param embed_size: Desired size out word embedding. Options
                       are 50, 100, 200, 300 based on GLoVe embeddings. 
    :return: A list of numpy arrays of shape M x N where M = embedding
             size, N = number of words in each essay.
             Note that N varies for each essay. 
    '''
    embed = load_word_embedding(embed_size=embed_size)

    num_essays = len(essays)
    print("The total number of essays is {}".format(num_essays))
    essay_vectorized = [None]*num_essays

    # Iterate through each essay and create the word embedding version
    for idx, e in enumerate(essays):
        # Convert individual essay from a string to a list of words
        e_words = essay_to_words(e, lowercase=True)

        # Convert an essay from a list of words to a matrix word
        # embedding representation.
        essay_vectorized[idx] = essay_to_wordembed(embed, e_words, embed_size)
        if (idx+1)%1000 == 0 and verbose==True:
            print("{} Essays Vectorized".format(idx+1))

    print("{} Total Essays Vectorized!".format(idx+1))

    return essay_vectorized

def essay_to_words(text, lowercase=True):
    '''
    This function separates an essay from a string into individual words
    :param text: Essay, as a string
    :param lowercase: True or False depending on if lowercase is desired
    :return: Essay as a list of words.
    '''

    # Use RegEx to separate essay into words. Note that special characters
    # are considered individual words.
    if lowercase == True:
        text_lower = text.lower()
        words = re.findall(r"[\w']+|[.,!?;]", text_lower)
    else:
        words = re.findall(r"[\w']+|[.,!?;]", text)

    return words

def unique_words(words):
    # *** Currently not used ***

    # Count word occurrences in the essay
    word_dict={}
    for word in words:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1

    # The list of unique words is a list of the keys in the word dictionary
    unique_words = [key for key in word_dict.keys()]
    
    return unique_words

def essay_to_wordembed(embed_dict, words, embed_size):
    '''
    This function will convert an essay in the form of a list of words
    to a matrix word embedding representation.
    :param  embed_dict: An embedding dictionary
    :param  words: A list of words in the essay
    :return: An matrix where each column represents a word in the essay (in 
            the order that they appear in the essay) and each row represents
            the word embedding values for that token. The matrix is M x N where
            M is the number of word vectors and N is the number of words in
            the essay.
    '''

    # Initialize embedding matrix
    num_words = len(words)
    embed_mat = np.zeros(shape=[embed_size, num_words], dtype='float32')
    for idx, word in enumerate(words):
        if word not in embed_dict:
            #print("{} not found in dictionary".format(word))
            # Unknown words have their own word embedding
            embed_mat[:,idx] = embed_dict['unk']
        else:
            embed_mat[:,idx] = embed_dict[word]

    return embed_mat

def load_word_embedding(embed_size):
    '''
    This function loads the relevant GLoVe word embedding and returns it
    as a dictionary.
    :param embed_size: Desired size of the word embedding
    :return: A dictionary representation of the word embedding. Each Key
             represents a word. Each Value represents a vector of corresponding
             coefficients to a given word. 
    '''
    # Read in correct word embedding file
    if embed_size == 50:
        embed_file = './data/glove.6B/glove.6B.50d.txt'
    elif embed_size == 100:
        embed_file = './data/glove.6B/glove.6B.100d.txt'
    elif embed_size == 200:
        embed_file = './data/glove.6B/glove.6B.200d.txt'
    elif embed_size == 300:
        embed_file = './data/glove.6B/glove.6B.300d.txt'

    # Store word embedding as a dictionary.
    # Keys: words, Values: Embedding vector
    with open(embed_file) as embed:
        embed_dict = {}
        for line in embed:
            line = line.split()
            word = line[0]
            embed_vec = np.array(line[1:], dtype='float32')
            embed_dict[word] = embed_vec

    return embed_dict



