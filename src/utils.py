import numpy as np

def get_batches(essays, scores, batch_size, net_type='lstm'):
    '''
    Batch generator to be used for training network
    :param essays: Numpy ndarray of essays. If net_type is 'lstm', the essays
                   are represented as n x m x e where n is the number of essays,
                   m is the number of words in each essay, and e is the size of
                   the word embedding.
    :param scores: Target values for each essay. Numpy ndarray of length n
    :param batch_size: Desired number of essays and corresponding labels to be 
                       output each loop of the generator.
    :param net_type: 'lstm', 'gru', or 'mlp' are supported
    :return: Yields a batch of essays as inputs and corresponding labels
    '''
    n_batches = int(len(essays) / batch_size)
    if net_type=='lstm' or net_type=='gru':
        while True:
            for i in range(n_batches):
                batch_X = essays[i * batch_size: (i + 1) * batch_size, :, :]
                batch_y = scores[i * batch_size: (i + 1) * batch_size]
                yield (batch_X, batch_y)

    elif net_type=='mlp':
        while True:
            essays_shuf, scores_shuf = shuffle(essays,scores)
            for i in range(n_batches):
                batch_X = essays_shuf[i * batch_size: (i + 1) * batch_size, :]
                batch_y = scores_shuf[i * batch_size: (i + 1) * batch_size]
                yield (batch_X, batch_y)

def shuffle(essays, scores):
    '''
    This function will shuffle a set of input essays and corresponding labels
    :param essays: Array of word embedded essays
    :param scores: Corresponding scores for each essay
    :return: A shuffled set of word embedded essays and corresponding labels
    '''
    # Create random mask and shuffle it
    num_essays = essays.shape[0]
    mask = np.arange(num_essays)
    mask = np.random.shuffle(mask)

    # Apply random mask to both essays and corresponding scores
    # Note: Shape differences between different network types
    if essays.ndim == 2:
        essays_shuffled = essays[mask,:]
    elif essays.ndim == 3:
        essays_shuffled = essays[mask,:,:]
    else:
        raise Exception('Input data shape unsupported.')

    scores_shuffled = scores[mask]

    return essays_shuffled[0], scores_shuffled[0]

def train_val_split(essays, scores, train_prop=0.8):
    '''
    This function splits input data and corresponding label values
    into training and validation sets according to a desired proportion
    :param essays: Essays in word embedded form
    :param scores: Scores associated with essays
    :param train_prop: Desired proportion of training data
    :return: X and y for both training and validation
    '''
    num_essays = essays.shape[0]
    num_train = int(np.ceil(num_essays*train_prop))
    num_val = int(num_essays - num_train)
    X_train = essays[:num_train, :]
    y_train = scores[:num_train]

    X_val = essays[-num_val:, :]
    y_val = scores[-num_val:]

    return X_train, y_train, X_val, y_val