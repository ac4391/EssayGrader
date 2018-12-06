import numpy as np

def get_batches(essays, scores, sets, batch_size, net_type='lstm'):
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
            batch_s = sets[i * batch_size: (i + 1) * batch_size]
            
            yield (batch_X, batch_y, batch_s)

    elif net_type=='mlp':
        while True:
            essays_shuf, scores_shuf, sets_shuf = shuffle(essays,scores, sets)
            for i in range(n_batches):
                batch_X = essays_shuf[i * batch_size: (i + 1) * batch_size, :]
                batch_y = scores_shuf[i * batch_size: (i + 1) * batch_size]
                batch_s = sets_shuf[i * batch_size: (i + 1) * batch_size]
                yield (batch_X, batch_y, batch_s)
                #print("Yielded range {} to {}".format(i*batch_size, (i+1)*batch_size))



def shuffle(essays, scores, sets):
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
    sets_shuffled = sets[mask]

    return essays_shuffled[0], scores_shuffled[0], sets_shuffled[0]

def train_val_split(essays, scores, sets, train_prop=0.8):
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
    print(sets)
    s_train = sets[:num_train]

    X_val = essays[-num_val:, :]
    y_val = scores[-num_val:]
    s_val = sets[-num_val:]

    return X_train, y_train, s_train, X_val, y_val, s_train

def normalize_predictions(predictions, dataset):
    for i,pred in (enumerate(predictions)):
        if (dataset[i] == 1):
            continue
        elif (dataset[i] == 2):
            predictions[i] = int(round(float(pred)/2))
        elif (dataset[i] == 3):
            predictions[i] = int(round(float(pred)/4))
        else:
            predictions[i] = int(round(float(pred)/3))
    return predictions

# SCRAPPED, USE ONLY FOR REFERENCE
def qwk(rater_a, rater_b, min_rating=None, max_rating=None):
    
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

# SCRAPPED, USE ONLY FOR REFERENCE
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat