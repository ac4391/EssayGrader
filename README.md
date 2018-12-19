# Automatic Essay Scoring with Neural Networks

This project is an attempt to generate accurate scores for standardized test essays by making use of 2 types of neural
networks: Multi-Layer Perceptron (MLP) and Recurrent Neural Network (RNN). The problem was proposed by the Hewlett
Foundation via a Kaggle competition [(Here)](https://www.kaggle.com/c/asap-aes) in 2012. Many of the solutions at the time
made use of traditional machine learning techniques and feature extraction by domain experts. In 2016, 
[Nguyen and Dery](https://cs224d.stanford.edu/reports/huyenn.pdf) attempted to tackle the same problem using Neural Networks.
They were able to successfully beat out the top machine learning approaches from the original competition.

This project is an attempt to both reproduce their results, and also improve upon them with new developments in the field
of Neural Networks, most notably Gated Recurrent Units (GRUs) as a replacement for Long-Short Term Memory (LSTM).

## Getting Started

Following these instructions will get you a copy of the project up and running on your local machine for development
and testing purposes.

### Prerequisites

Running the code in this repository requires elementary knowledge of both Jupyter and Anaconda. It is recommended that 
new users create a new virtual environment with Anaconda to ensure that package dependencies match the developer 
versions. If you are unfamiliar with Anaconda, you can find more information and getting started tutorials here:
https://conda.io/docs/user-guide/overview.html

Note that python version 3.6.7 was used for this project. To create a new Anaconda environment, you may use the terminal
command:
```
conda create -n name_of_myenv python=3.6.7
```
After creating this environment, you may clone this repository to your local machine. Within the top level directory,
you will find a 'req.txt' file, which includes a comprehensive list of dependencies necessary to execute the functionality
of this repository. With your new environment active, use the following command to install these dependencies:
```
pip install -r /path/to/req.txt
```

## Running the Notebooks

Now that the necessary packages are installed, you can move to the 'preprocess.ipynb' jupyter notebook. This notebook
takes the raw essay text (from the data file data/training_set_rel3.tsv) and completes necessary preprocessing to prepare
the data for input to a neural network. If you do not have this dataset, download it from the link 
[(Here)](https://www.kaggle.com/c/asap-aes/data) 

To run this notebook, you may run the following command:
```
jupyter notebook 
```
and navigate to the preprocess.ipynb file.

After running the preprocess.ipynb notebook, two large data files 'train_df.pkl' and 'test_df.pkl' will be saved within the
'data/' directory on the user's local machine. This contains the preprocessed data to be loaded as input
to the Neural Network models. 

The next step is to run the main Jupyter notebook: 'EssayGrader.ipynb'. This notebook contains the main functionality
for the project including running the various Neural Networks and interpreting the results. 

When Running 'EssayGrader.ipynb' you will produce a model based on the desired parameters. This model is then used
to create predictions based on a test set. The QWK values are calculated based on these predictions.


## Results

The QWK value measures the difference between the true grade of the essay and the predicted grade. 
A value of 1 corresponds to a complete match, and a value of 0 corresponds to no improvement over random guessing.

Using MLP, LSTM, and GRU network architectures, the best QWK score attained was 0.74. 

![Kappa Scores](data/kappa_scores.png)



## Built With

* [Tensorflow](https://www.tensorflow.org/) - The Neural Network framework used
* [NumPy](http://www.numpy.org/) - Matrix operations and linear algebra
* [Pandas](https://pandas.pydata.org/) - Data preparation
* [GloVe](https://nlp.stanford.edu/projects/glove/) - Word Embeddings for vector representations for words
* [Seaborn](https://seaborn.pydata.org/) - Data visualization


## Authors

* **Brian Midei** - [bmmidei](https://github.com/bmmidei)
* **Ariel Cohen-Codar** - [ac4391](https://github.com/ac4391)
* **Marko Mandic** - [markomandic](https://github.com/markomandic)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Kaggle competition for posing the original problem and providing data - https://www.kaggle.com/c/asap-aes
* Nguyen and Dery for writing the original paper, inspiring us to develop our own model to reproduce/improve upon results.
    * Link to original paper - https://cs224d.stanford.edu/reports/huyenn.pdf
* ECBM4040 - Professor Kostic and TAs for advice to get the model up and running
