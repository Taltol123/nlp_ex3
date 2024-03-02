import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
RARE = "rare"
NEGATED = "negated"

np.random.seed(0)
torch.manual_seed(0)
# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# TODO: when we need to use this function?
def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v_average = np.zeros(embedding_dim)

    for word in sent.text:
        vector_to_add = word_to_vec.get(word, np.zeros(300))
        w2v_average = np.add(w2v_average, vector_to_add)

    return w2v_average / len(sent.text)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1

    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vocabulary_size = len(word_to_ind)
    one_hots_average = np.zeros(vocabulary_size)

    for word in sent.text:
        one_hots_average = np.add(one_hots_average, get_one_hot(vocabulary_size, word_to_ind[word]))

    return one_hots_average / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    mapping = {}
    index = 0
    for word in words_list:
        if word not in mapping:
            mapping[word] = index
            index += 1
    return mapping


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    arr = np.zeros((seq_len, embedding_dim))
    sent_len = len(sent.text)
    if sent_len < seq_len:
        for i, word in enumerate(sent.text):
            if word in word_to_vec:
                arr[i] = word_to_vec.get(word)
    else:
        sent.text = sent.text[:seq_len]
        for i, word in enumerate(sent.text):
            if word in word_to_vec:
                arr[i] = word_to_vec.get(word)
    return arr


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()
        rare_word_indexs = data_loader.get_rare_words_examples(
            self.sentiment_dataset.get_test_set(), self.sentiment_dataset)
        rare_word_arr = []
        for idx in rare_word_indexs:
            rare_word_arr.append(self.sentences[TEST][idx])
        self.sentences[RARE] = rare_word_arr

        negate_word_indexs = data_loader.get_negated_polarity_examples(
            self.sentiment_dataset.get_test_set())
        negated_word_arr = []
        for idx in negate_word_indexs:
            negated_word_arr.append(self.sentences[TEST][idx])
        self.sentences[NEGATED] = negated_word_arr

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this (only of x,
        ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text):
        text = text.to(torch.float32)
        lstm_run_out, lstm_hidden_and_cell = self.lstm(text)
        lstm_hidden = lstm_hidden_and_cell[0]
        h_n_forward = lstm_hidden[-2, :, :]
        h_n_backward = lstm_hidden[-1, :, :]
        lstm_out = torch.cat((h_n_forward, h_n_backward), dim=1)
        lstm_out = self.dropout(lstm_out)
        return self.linear_layer(lstm_out).squeeze()

    def predict(self, text):
        return torch.sigmoid(self(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1, bias=True)
        get_available_device()

    def forward(self, x):
        x = x.to(torch.float32)
        return self.linear(x)

    def predict(self, x):
        return nn.functional.sigmoid(self(x))  # because we are using sigmoid
        # only in the train before that and we want to return a number
        # between 0 to 1


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    counter = 0.0
    for i in range(len(preds)):
        if preds[i] == y[i]:
            counter += 1

    return counter / len(preds)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for
    the model. - a dict that have train validation and test with the needed
    data loaders
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()
    for batch in data_iterator:
        x, y = batch[0], batch[1]
        y = torch.tensor(y)
        y = y.view(x.shape[0], 1)
        # x.shape == (batch_size,embedding_dim) if we had the dim
        # y.shape == (batch,size,1)
        y_predicted = model(x)
        y_predicted = y_predicted.view(x.shape[0], 1) # I added now

        loss = criterion(y_predicted, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models.
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    sum = 0
    accuracy = 0
    counter = 0

    model.eval()
    with torch.no_grad():
        for batch in data_iterator:
            x, y = batch[0], batch[1]  # this is from get_item of OnlineDataset
            # x.shape == (batch_size,embedding_dim) if we had the dim
            # y.shape == (batch,size,1) 1 cause of the output size
            y = torch.tensor(y)
            y = y.view(x.shape[0], 1)
            y_predicted = model.predict(x)
            y_predicted = y_predicted.view(x.shape[0], 1) #I added now
            sum += criterion(y_predicted, y).item()
            y_predicted = (y_predicted > 0.5)
            y_predicted = [1 if x else 0 for x in y_predicted]
            accuracy += binary_accuracy(y_predicted, y)
            counter += data_iterator.batch_size

    # return sum / counter, accuracy / counter
    return sum / len(data_iterator), accuracy / len(data_iterator)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    return model.predict(data_iter)


# def create_charts(model_name, accuracy_train, loss_train, accuracy_validate, loss_validate):
#     # Extracting epoch numbers and loss values for train and validation sets
#     train_epochs = list(loss_train.keys())
#     train_losses = list(loss_train.values())
#
#     validation_epochs = list(loss_validate.keys())
#     validation_losses = list(loss_validate.values())
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_epochs, train_losses, label='Train Loss', marker='o')
#     plt.plot(validation_epochs, validation_losses, label='Validation Loss', marker='s')
#
#     # Adding labels and title
#     plt.title('Train and Validation Loss over Epochs')
#     plt.xlabel('Epoch Number')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     # Displaying the plot
#     plt.grid(True)
#     plt.show()
#
#     # Accuracy
#     train_epochs = list(accuracy_train.keys())
#     train_losses = list(accuracy_train.values())
#
#     validation_epochs = list(accuracy_validate.keys())
#     validation_losses = list(accuracy_validate.values())
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_epochs, train_losses, label='Train Accuracy', marker='o')
#     plt.plot(validation_epochs, validation_losses, label='Validation Accuracy', marker='s')
#
#     # Adding labels and title
#     plt.title('Train and Validation Accuracy over Epochs')
#     plt.xlabel('Epoch Number')
#     plt.ylabel('Accuracy')
#     plt.legend()
#
#     # Displaying the plot
#     plt.grid(True)
#     plt.show()


def create_charts(model_name, accuracy_train, loss_train, accuracy_validate, loss_validate):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Create subplots with 1 row and 2 columns

    # Plotting loss
    axs[0].plot(list(loss_train.keys()), list(loss_train.values()), label='Train Loss', marker='o')
    axs[0].plot(list(loss_validate.keys()), list(loss_validate.values()), label='Validation Loss', marker='s')
    axs[0].set_title('Train and Validation Loss over Epochs')
    axs[0].set_xlabel('Epoch Number')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting accuracy
    axs[1].plot(list(accuracy_train.keys()), list(accuracy_train.values()), label='Train Accuracy', marker='o')
    axs[1].plot(list(accuracy_validate.keys()), list(accuracy_validate.values()), label='Validation Accuracy',
                marker='s')
    axs[1].set_title('Train and Validation Accuracy over Epochs')
    axs[1].set_xlabel('Epoch Number')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(f'Model: {model_name}')  # Adding main title with model_name
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


# Example usage:
# create_charts(model_name, accuracy_train, loss_train, accuracy_validate, loss_validate)


def train_model(model_name, model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """

    data_iterator_train = data_manager.get_torch_iterator()
    data_iterator_val = data_manager.get_torch_iterator(VAL)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    accuracy_train = {}
    loss_train = {}
    accuracy_validate = {}
    loss_validate = {}

    for i in tqdm(range(n_epochs), desc='Training progress'):
        train_epoch(model, data_iterator_train, optimizer, criterion)
        train_pair = evaluate(model, data_iterator_train, criterion)
        loss_train[i] = train_pair[0]
        accuracy_train[i] = train_pair[1]
        validate_pair = evaluate(model, data_iterator_val, criterion)
        loss_validate[i] = validate_pair[0]
        accuracy_validate[i] = validate_pair[1]
    # for i in range(n_epochs):
    #         train_epoch(model, data_iterator_train, optimizer, criterion)
    #         train_pair = evaluate(model, data_iterator_train, criterion)
    #         loss_train[i] = train_pair[0]
    #         accuracy_train[i] = train_pair[1]
    #         validate_pair = evaluate(model, data_iterator_val, criterion)
    #         loss_validate[i] = validate_pair[0]
    #         accuracy_validate[i] = validate_pair[1]
    create_charts(model_name, accuracy_train, loss_train, accuracy_validate, loss_validate)

    return


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(batch_size=64)
    embedding_dim = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_dim)
    learning_rate = 0.01
    epochs = 20
    weight_decay = 0.001
    train_model("Simple log-linear model", model, data_manager, epochs, learning_rate, weight_decay)
    return model, data_manager


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(batch_size=64, data_type=W2V_AVERAGE, embedding_dim=300)
    embedding_dim = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_dim)
    learning_rate = 0.01
    epochs = 20
    weight_decay = 0.001
    train_model("Word2Vec log-linear model", model, data_manager, epochs, learning_rate, weight_decay)
    return model, data_manager


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(batch_size=64, data_type=W2V_SEQUENCE, embedding_dim=300)
    embedding_dim = data_manager.get_input_shape()[1]
    learning_rate = 0.001
    epochs = 4
    dropout = 0.5
    weight_decay = 0.0001
    model = LSTM(embedding_dim, hidden_dim=100, n_layers=1, dropout=dropout)
    train_model("LSTM model", model, data_manager, epochs, learning_rate, weight_decay)
    return model, data_manager


def check_testset_log_linear(model, data_manager):
    test_iterator = data_manager.get_torch_iterator(TEST)
    criterion = nn.BCEWithLogitsLoss()
    results = evaluate(model, test_iterator, criterion)
    print("ONE HOT Accuracy: ")
    print(results[1])
    print("ONE HOT Loss: ")
    print(results[0])
    rare_results = evaluate(model, data_manager.get_torch_iterator(RARE),
                            criterion)
    negated_results = evaluate(model, data_manager.get_torch_iterator(
        NEGATED), criterion)
    print("ONE HOT Rare Words Accuracy:")
    print(rare_results[1])
    print("ONE HOT Negated Words Accuracy:")
    print(negated_results[1])


def check_testset_log_linear_w2v_avg(model, data_manager):
    test_iterator = data_manager.get_torch_iterator(TEST)
    criterion = nn.BCEWithLogitsLoss()
    results = evaluate(model, test_iterator, criterion)
    print("W2V AVG Accuracy: ")
    print(results[1])
    print("W2V AVG Loss: ")
    print(results[0])
    rare_results = evaluate(model, data_manager.get_torch_iterator(RARE),criterion)
    negated_results = evaluate(model, data_manager.get_torch_iterator(
        NEGATED), criterion)
    print("W2V Rare Words Accuracy:")
    print(rare_results[1])
    print("W2V Negated Words Accuracy:")
    print(negated_results[1])


def check_testset_lstm_w2v(model, data_manager):
    results = evaluate(model, data_manager.get_torch_iterator(TEST),
                       nn.BCEWithLogitsLoss())
    print("LSTM Accuracy: ")
    print(results[1])
    print("LSTM Loss: ")
    print(results[0])
    rare_results = evaluate(model, data_manager.get_torch_iterator(RARE),
                            nn.BCEWithLogitsLoss())
    negated_results = evaluate(model, data_manager.get_torch_iterator(
        NEGATED), nn.BCEWithLogitsLoss())
    print("LSTM Rare Words Accuracy:")
    print(rare_results[1])
    print("LSTM Negated Words Accuracy:")
    print(negated_results[1])


if __name__ == '__main__':
    # train_log_linear_with_one_hot_res = train_log_linear_with_one_hot()
    # one_hot_model = train_log_linear_with_one_hot_res[0]
    # one_hot_data_manager = train_log_linear_with_one_hot_res[1]
    # check_testset_log_linear(one_hot_model, one_hot_data_manager)
    #
    # train_log_linear_with_w2v_res = train_log_linear_with_w2v()
    # w2v_avg_model = train_log_linear_with_w2v_res[0]
    # w2v_avg_data_manager = train_log_linear_with_w2v_res[1]
    # check_testset_log_linear_w2v_avg(w2v_avg_model, w2v_avg_data_manager)

    train_lstm_with_w2v_res = train_lstm_with_w2v()
    lstm_model = train_lstm_with_w2v_res[0]
    lstm_data_manager = train_lstm_with_w2v_res[1]
    check_testset_lstm_w2v(lstm_model, lstm_data_manager)
