import numpy as np
import plotly
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='pmt210', api_key='z98OIeWqz9ee8yG8tCN9')

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
            # Shuffle essays after each full pass over the data
            essays_shuf, scores_shuf = shuffle(essays, scores)
            for i in range(n_batches):
                batch_X = essays_shuf[i * batch_size: (i + 1) * batch_size, :, :]
                batch_y = scores_shuf[i * batch_size: (i + 1) * batch_size]
                yield (batch_X, batch_y)

    elif net_type=='mlp':
        while True:
            # Shuffle essays after each full pass over the data
            essays_shuf, scores_shuf = shuffle(essays, scores)
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
    np.random.shuffle(mask)

    # Apply random mask to both essays and corresponding scores
    # Note: Shape differences between different network types
    if essays.ndim == 2:
        essays_shuffled = essays[mask,:]
    elif essays.ndim == 3:
        essays_shuffled = essays[mask,:,:]
    else:
        raise Exception('Input data shape unsupported.')
    scores_shuffled = scores[mask]

    return essays_shuffled, scores_shuffled

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

def preds_to_scores(preds, min_score):
    '''
    This function shifts network predictions in a 0-based scale to a
    desired scale
    :param preds: array of predictions from the network
    :param min_score: desired lowest value of the output scaled data
    :return: Array of shifted scores
    '''
    return np.array([pred+min_score for pred in preds])

def scores_to_preds(scores, min_score):
    '''
    Corollary to preds_to_scores. Shifts score labels to 0-based
    :param scores: array of essay scores
    :param min_score: lowest score of input data scale
    :return: Array of shifted scores
    '''
    return np.array([score-min_score for score in scores])

def normalize_predictions(predictions, dataset):
    '''
    Normalize scores from various essay sets to a single scale
    :param predictions: Essay score predictions
    :param dataset: Input dataset
    :return: Normalized predictions
    '''
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








def confusion_matrix(score1, score2, num_classes):
    '''
    Matrix mat of size num_classes x num_classes
    mat[i][j] is the numbers of time rater1 gave i and rater2 gave j
    '''
    assert(len(score1) == len(score2))
    conf_mat = [[0 for i in range(num_classes)]
                for j in range(num_classes)]
    for a, b in zip(score1, score2):
        conf_mat[a][b] += 1
    return conf_mat

def histogram(scores, num_classes):
    '''
    Distribution of the scores
    :param scores: list of scores
    :param num_classes: range of scores possible
    :return: Distribution of the scores
    '''
    hist_scores = [0 for x in range(num_classes)]
    for r in scores:
        hist_scores[r] += 1
    return hist_scores

def quadratic_weighted_kappa(score1, score2, num_classes):
    '''
    Kappa Statistic mesure the degree of randomness (no correlation) between two range of scores
    It evaluates how coherent are the scores of each rater (or with our prediction)
    kappa=0 if random and 1 if the raters agree on the same score.
    :param score1: first list of scores
    :param score2: second list of scores
    :param num_classes: range of scores possible
    :return: performance metric quadratic weighted kappa
    '''
    assert(len(score1) == len(score2))
    conf_mat = confusion_matrix(score1, score2, num_classes)
    num_scores = len(conf_mat)
    num_scored_items = float(len(score1))

    hist_score1 = histogram(score1, num_classes)
    hist_score2 = histogram(score2, num_classes)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_scores):
        for j in range(num_scores):
            expected_count = (hist_score1[i] * hist_score2[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_scores - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator










def plot_train_loss(train_loss_hist, val_loss_hist, n_batches, model_name):
    '''
    This function will plot the training and validation loss throughout the 
    training process.
    :param train_loss_hist: training loss history
    :param val_loss_hist: validation loss history
    :param n_batches: Number of batches per training epoch
    :return: plotly figure
    '''
    # extract data from training and validation loss history
    x = []
    train_loss = []
    val_loss = []
    for k,v in train_loss_hist.items():
        x.append(k[0]-1+k[1]/n_batches)
        train_loss.append(v)
        val_loss.append(val_loss_hist[k])
        
    # define plotly figure parameters 
    Train_loss = go.Scatter(
        x=x,
        y=train_loss,
        name='Training Loss')
    Val_loss = go.Scatter(
        x=x,
        y=val_loss,
        name='Validation Loss')
    layout= go.Layout(
        title= 'Training and Validation loss for model: {}'.format(model_name),
        hovermode= 'closest',
        xaxis= dict(
            title= 'Epochs',
            ticklen= 5,
            zeroline= False,
            gridwidth= 2,
        ),
        yaxis=dict(
            title= 'Loss',
            ticklen= 5,
            gridwidth= 2,
        ),
        showlegend= True
    )
    data = [Train_loss, Val_loss]
    fig= go.Figure(data=data, layout=layout)
    return fig
        
    