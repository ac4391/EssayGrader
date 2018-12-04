import numpy as np

def get_batches(essays, scores, batch_size, net_type='lstm'):
    '''
    Not implemented fully. This just a start.
    *** Not functional ***
    :param essays: 
    :param scores: 
    :param n_essays: 
    :param net_type: 
    :return: 
    '''
    n_batches = int(len(essays) / batch_size)
    if net_type=='lstm' or net_type=='gru':
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
                #print("Yielded range {} to {}".format(i*batch_size, (i+1)*batch_size))



def shuffle(essays, scores):
    '''
    This function will shuffle a set of input data and corresponding labels
    :param essays: Array of word embedded essays
    :param scores: Corresponding scores for each essay
    :return: A shuffled set of word embedded essays and corresponding labels
    '''
    # Create random mask
    num_essays = essays.shape[0]
    mask = np.arange(num_essays)

    mask = np.random.shuffle(mask)

    # Apply random mask to both essays and corresponding scores
    essays_shuffled = essays[mask,:]
    scores_shuffled = scores[mask]

    return essays_shuffled[0], scores_shuffled[0]

def train_val_split(essays, scores, train_prop=0.8):
    '''
    This function splits input data and corresponding ground truth values
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