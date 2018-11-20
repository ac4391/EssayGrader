import numpy as np
import codecs

# download the data
def get_data(data):
    features = []
    # read file line by line
    data_file = codecs.open(data, encoding='utf-8')
    lines = data_file.readlines()
    data_file.close()
    num_lines = len(lines)
    line = 0
    for one_line in lines:
        row = one_line.split('\n')[0].split('\t')
        vector = []
        # Ignore the heading line
        if line < 1:
            line += 1
            continue

        # treat each line as an essay
        e = Essay(row, store_score = True)
        # extract features
        f = e.features
        for i in sorted(f.__dict__.keys()):
            vector.append(f.__dict__[i])
        # append to feature vector
        vector.append(e.score)
        features.append(np.array(vector))
        line += 1
    return np.array(features)
