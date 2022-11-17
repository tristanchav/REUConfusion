import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import seaborn as sns
import os
import csv
import pandas as pd
import pickle

subjects = []
TIMESTAMP_CSV = r'Timestamps.csv'  # directory containing timestamps related to each session
TIMES_CSV = r'Times.csv'  # directory containing timestamps converted into seconds
SUBJECT_DIR = r'subjects'  # directory containing subject csvs
BUILDER_DIR = r'Builder_Data'  # directory containing subject freemocap
INSTRUCTOR_DIR = r'Instructor_Data'  # directory containing instructor subject data
LABELS_CSV = r'Confuseometer_Outputs.csv'  # directory containing confusion labels
EPOCHS = 250     # Number of epochs/iterations to be ran over the data
HIDDENLAYERS = 128   # Number of nodes in the hidden layer
LR = .001   # The learning rate
WINDOWSIZE = 150   # The number of seconds in each window


# -----------------------------------------------------PICKLING---------------------------------------------------------
# Pickles the RNN class and saves the accuracy, loss and window size of the lowest loss recorded to a txt file
def pickling(RNN, accuracy, bestLoss):
    f = open("RNNSaved", 'wb')
    pickle.dump(RNN, f)
    with open("output.txt", "w") as of:
        of.write("Accuracy: " + str(accuracy) + "\nBest Loss: " + str(bestLoss) + "\nWindow Size: " + str(WINDOWSIZE))
        of.close()
    f.close()

# Unpickles the RNN class
def unpickle():
    f = open('RNNSaved', 'rb')
    rnn = pickle.load(f)
    f.close()
    return rnn


# -------------------------------------------------RNN-MODEL------------------------------------------------------------
# The class which has the RNN variables and class-specific functions
class RNN:
    '''
    Implements a recurrent neural network with predict, train, and validate methods
    '''

    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        Sets network parameters including initialization of network dimensions,
        initializes random W, V, and U matrices, sets the activation function to tanh,
        and defines the number of epochs and learning rate.
        Inputs:
        input_dim: dimension of features in dataset
        hidden_dim: hidden dimension of choice
        output_dim: number of categories network is trying to predict
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scale = 0.5
        W = np.random.normal(0, 1, (input_dim, hidden_dim)) * self.scale
        V = np.random.normal(0, 1, (hidden_dim, hidden_dim)) * self.scale
        U = np.random.normal(0, 1, (hidden_dim, output_dim)) * self.scale
        W = tf.Variable(tf.convert_to_tensor(W, dtype=tf.float64))
        V = tf.Variable(tf.convert_to_tensor(V, dtype=tf.float64))
        U = tf.Variable(tf.convert_to_tensor(U, dtype=tf.float64))
        b = np.zeros((1, self.hidden_dim))
        c = np.zeros((1, self.output_dim))
        b = tf.Variable(tf.convert_to_tensor(b, dtype=tf.float64))
        c = tf.Variable(tf.convert_to_tensor(c, dtype=tf.float64))

        self.theta = [W, V, U, b, c]
        self.activation = tf.nn.softmax
        self.num_epoch = EPOCHS
        self.learning_rate = LR
        self.cce = tf.keras.losses.CategoricalCrossentropy()

    def train(self, X, Y, test_X, test_Y):
        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []
        optimizer = Adam(learning_rate=self.learning_rate, decay=self.learning_rate / self.num_epoch)

        # iterate over timesteps in sequence
        bestLoss = 10000
        for r in range(self.num_epoch):
            print("Epoch: " + str(r))
            loss = 0.0  # reset loss
            # randomly order participants so we see them in a different order on each training iteration
            randomly_ordered_indices = np.random.permutation([i for i in range(len(X))])
            preds = []
            actuals = []
            for i in randomly_ordered_indices:
                x, y = X[i], Y[i]
                with tf.GradientTape() as tape:
                    tape.watch(self.theta)
                    prev_activations = tf.convert_to_tensor(np.zeros((1, self.hidden_dim)),
                                                            dtype=tf.float64)  # begins as all zeros
                    pred = None
                    for t in range(x.shape[0]):
                        next_input = x[t, :]
                        next_input = tf.expand_dims(next_input, axis=0)
                        WdotNew = tf.matmul(next_input, self.theta[0])
                        VdotPrevious = tf.matmul(prev_activations, self.theta[1])
                        add = WdotNew + VdotPrevious + self.theta[3]
                        h = self.activation(add)
                        UdotS = tf.nn.softmax(tf.matmul(h, self.theta[2]) + self.theta[4])
                        pred = UdotS
                        # print(pred[0].numpy())
                        prev_activations = h
                        # print(pred)

                        pred = tf.cast(pred, tf.float32)
                        next_label = y[t, :]
                        # print(prediction[0].numpy())
                        # print(next_label.numpy())
                        # print(next_label)
                        # print(pred[0])
                        individual_loss = self.cce(next_label, pred[0])
                        loss = individual_loss + loss
                        preds.append(pred[0].numpy())
                        actuals.append(y[t])
                    gradients = tape.gradient(loss, self.theta)  # gradients = [dW, dV, dU]
                    optimizer.apply_gradients(zip(gradients, self.theta))
            # loss = loss / len(randomly_ordered_indices)
            print("Training loss:", loss.numpy() / len(X))
            training_losses.append(loss.numpy() / len(X))

            validation_loss, validation_accuracy = self.validate(test_X, test_Y)

            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            discrete_preds = []
            for prediction in preds:
                discrete_preds.append(tf.math.argmax(prediction))

            discrete_actuals = []
            for actual in actuals:
                discrete_actuals.append(tf.math.argmax(actual))

            binned_preds, binned_actuals = [], []
            for pred in discrete_preds:
                if pred.numpy() == 0 or pred.numpy() == 1:
                    binned_preds.append(0)
                else:
                    binned_preds.append(1)

            for actual in discrete_actuals:
                # print(actual.numpy())
                if actual.numpy() == 0 or actual.numpy() == 1:
                    binned_actuals.append(0)
                else:
                    binned_actuals.append(1)

            m = tf.keras.metrics.Accuracy()
            m.update_state(binned_preds, binned_actuals)
            training_accuracies.append(m.result().numpy())

            print("Validation loss:", validation_loss)

            if validation_loss < bestLoss:
                bestLoss = validation_loss
                pickling(self, m.result().numpy(), bestLoss)

        # print(self.theta)
        return training_losses, validation_losses, training_accuracies, validation_accuracies

    # Validation with validation set is done here
    def validate(self, X, Y):
        # print(X.shape)
        loss = 0.0  # reset loss
        preds = []
        actuals = []
        # randomly order participants so we see them in a different order on each training iteration
        randomly_ordered_indices = np.random.permutation([i for i in range(X.shape[0])])
        # print(randomly_ordered_indices.shape)
        for i in randomly_ordered_indices:
            x, y = X[i], Y[i]
            prev_activations = tf.convert_to_tensor(np.zeros((1, self.hidden_dim)),
                                                    dtype=tf.float64)  # begins as all zeros
            pred = None
            for t in range(x.shape[0]):
                next_input = x[t, :]
                next_input = tf.expand_dims(next_input, axis=0)
                WdotNew = tf.matmul(next_input, self.theta[0])
                VdotPrevious = tf.matmul(prev_activations, self.theta[1])
                add = WdotNew + VdotPrevious
                h = self.activation(add)
                UdotS = tf.nn.softmax(tf.matmul(h, self.theta[2]))
                pred = UdotS
                # print(pred[0].numpy())
                prev_activations = h
                # print(pred)
                pred = tf.cast(pred, tf.float32)
                next_label = y[t, :]
                individual_loss = self.cce(next_label, pred[0])
                loss = individual_loss + loss
                preds.append(pred[0].numpy())
                actuals.append(y[t])
            # loss = loss / len(randomly_ordered_indices)
        # print(loss.numpy() / len(X))
        loss = loss.numpy() / len(X)

        discrete_preds = []
        for prediction in preds:
            discrete_preds.append(tf.math.argmax(prediction))

        discrete_actuals = []
        for actual in actuals:
            discrete_actuals.append(tf.math.argmax(actual))

        binned_preds, binned_actuals = [], []
        for pred in discrete_preds:
            # print(pred.numpy())
            if pred.numpy() == 0 or pred.numpy() == 1:
                binned_preds.append(1)
            else:
                binned_preds.append(0)

        for actual in discrete_actuals:
            # print(actual.numpy())
            if actual.numpy() == 0 or actual.numpy() == 1:
                binned_actuals.append(1)
            else:
                binned_actuals.append(0)

        # re-enable outer spines
        matrix = confusion_matrix(binned_preds, binned_actuals)
        print(matrix)
        display = ConfusionMatrixDisplay(matrix, display_labels=["Confused", "Not Confused"])
        display.plot(colorbar=False, cmap=plt.cm.Blues)
        plt.show()

        sns.heatmap(matrix, fmt='d', annot=True, square=True,
                    cmap='gray_r', vmin=0, vmax=0,  # set all to white
                    linewidths=1, linecolor='k',  # draw black grid lines
                    cbar=False, xticklabels=["Confused", "Not Confused"], yticklabels=["Confused", "Not Confused"])
        sns.despine(left=False, right=False, top=False, bottom=False)
        sns.set(font_scale=6)
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.tight_layout()
        plt.show()

        m = tf.keras.metrics.Accuracy()
        m.update_state(binned_preds, binned_actuals)
        validation_accuracy = (m.result().numpy())

        return loss, validation_accuracy


# Runs the RNN and breaks up the sequences into training, validation and training sets
def Run_RNN(sequences, labels, final=False):
    rnn = RNN(input_dim=sequences.shape[2], hidden_dim=HIDDENLAYERS, output_dim=labels.shape[2])  # initialize RNN

    valid_sequence_length = int(0.1 * sequences.shape[0])
    test_sequence_length = int(0.9 * sequences.shape[0])
    train_sequence_length = sequences.shape[0] - valid_sequence_length - test_sequence_length

    # print(test_sequence_length)
    test_sequence = sequences[0:test_sequence_length]
    test_labels = labels[0:test_sequence_length]
    # print(len(test_sequence))

    # Prepare training and testing data by randomly selecting participants and separating them into test and train sets
    randomly_ordered_indices = np.random.permutation([i for i in range(test_sequence_length, sequences.shape[0])])
    test_sequence, test_labels = windowing(test_sequence, test_labels)
    print("Shape of Testing Set", test_sequence.shape)
    print("Shape of Testing Labels", test_labels.shape)

    if final:
        test_sequence = tf.cast(test_sequence, tf.float64)
        test_labels = tf.cast(test_labels, tf.float32)
        finalRNN = unpickle()
        print("Test Set Accuracy: ", finalRNN.validate(test_sequence, test_labels)[1])

    else:
        # Train Set
        train_sequence = []
        train_labels = []
        train_sequence_indices = randomly_ordered_indices[:train_sequence_length]
        # print(train_sequence_indices)
        for index in train_sequence_indices:
            train_sequence.append(sequences[index])
            train_labels.append(labels[index])
            # print(index)
            # print(sequences[index])
            # print(labels[index])

        train_sequence = tf.convert_to_tensor(train_sequence, dtype=tf.float64)
        train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
        train_sequence, train_labels = windowing(train_sequence, train_labels)

        print("Shape of Training Set", train_sequence.shape)
        print("Shape of Training Labels", train_labels.shape)

        # Test Set
        valid_sequence = []
        valid_labels = []
        valid_sequence_indices = randomly_ordered_indices[train_sequence_length:]
        # print(valid_sequence_indices)
        for index in valid_sequence_indices:
            valid_sequence.append(sequences[index])
            valid_labels.append(labels[index])
            # print(index)
            # print(sequences[index])
            # print(labels[index])

        valid_sequence = tf.convert_to_tensor(valid_sequence, dtype=tf.float64)
        valid_labels = tf.convert_to_tensor(valid_labels, dtype=tf.float32)
        valid_sequence, valid_labels = windowing(valid_sequence, valid_labels)

        print("Shape of Validation Set", valid_sequence.shape)
        print("Shape of Validation Labels", valid_labels.shape)

        training_losses, validation_losses, training_accuracies, validation_accuracies = \
            rnn.train(train_sequence, train_labels, valid_sequence, valid_labels)

        plt.plot([i for i in range(rnn.num_epoch)], training_losses)
        plt.plot([i for i in range(rnn.num_epoch)], validation_losses)
        plt.legend(['training', 'validation'], loc='upper right')
        plt.title("Training and Validation Loss for ")
        plt.xlabel("Number of epochs elapsed")
        plt.ylabel("Loss")
        plt.show()

        plt.plot([i for i in range(rnn.num_epoch)], training_accuracies, validation_accuracies)
        plt.legend(['training', 'validation'], loc='upper right')
        plt.title("Training and Validation Accuracy for RNN")
        plt.xlabel("Number of epochs elapsed")
        plt.ylabel("Accuracy")
        plt.show()

        # print("Validation Accuracy:", validation_accuracies[-1])


# -------------------------------------------------FREEMOCAP-TENSORS----------------------------------------------------

# Definition of function which returns the number of builders based on the number of folders in the Builder directory
def get_num_builders():
    builders = []  # list to contain builder names
    num_builders = 0  # initialized variable to hold the number of subject information
    # Establishes number of subjects by checking how many files are in the subjects subdirectory and extract their names
    for path in os.listdir(BUILDER_DIR):
        builders.append(path)  # Adds name of subject to subjects list
        num_builders += 1

    # print('Number of Builders:', num_builders)
    # print('Builder Names: ', builders)
    subjects.extend(builders)
    return num_builders

# Returns the maximum number of frames in freemocap data after converting to 1 frame per second
def get_max_frames():
    max_frames = 0
    for path in os.listdir(BUILDER_DIR):
        row = np.load(os.path.join(BUILDER_DIR, os.path.join(path, 'mediaPipeSkel_3d.npy')))
        frames = row.shape[0]
        if path == 'B3C' or path == 'B9C':
            frames = int(frames / 28.7)
        else:
            frames = int(frames / 15)
        # print(frames) debug
        if frames > max_frames:
            max_frames = frames
    return max_frames


# Creates a tensor for the freeMocap subjects
def FreeMocapTensor():
    max_frames = get_max_frames()
    # print(max_frames) debug
    num_subjects = get_num_builders()
    z_counter = 0
    subject_data_list = [[0]] * num_subjects
    for path in os.listdir(BUILDER_DIR):

        # Gets frame to start at from FreeMocap_Times.csv
        with open('FreeMocap_Times.csv', newline='') as csvfile:
            reader = list(csv.reader(csvfile))
            # print(int(path[1:-1])) debug
            startFrame = int(reader[int(path[1:-1])][1])
            # print(startFrame) debug

        row = np.load(os.path.join(BUILDER_DIR, os.path.join(path, 'mediaPipeSkel_3d.npy')))
        y_counter = 0
        frame_to_get = startFrame
        frame_data = np.zeros(shape=(max_frames, row.shape[1] * row.shape[2]))

        # Flattens all data for the specific timestamp
        frames = 0
        if path == 'B3C' or path == 'B9C':
            frames = int(row.shape[0] / 28.7) - int(startFrame / 28.7)
        else:
            frames = int(row.shape[0] / 15) - int(startFrame / 15)

        while y_counter < frames:
            # print(y_counter)
            # print(frames)

            if path == 'B3C' or path == 'B9C':
                frame_data[y_counter] = row[frame_to_get].flatten()
                y_counter += 1
                frame_to_get = int(frame_to_get + 28.7)
            else:
                frame_data[y_counter] = row[frame_to_get].flatten()
                y_counter += 1
                frame_to_get = frame_to_get + 15

            # if path == 'B2N':
            # print(frame_data[0])
            # print(int(path[1:-1])-1)
            # print(row[(y_counter-1) * 15].flatten().tolist())
            # print(y_counter - 1)
            # print(frames) debug
            # print(frame_data[y_counter-1])

        subject_data_list[int(path[1:-1]) - 1] = frame_data
        # print(subject_data_list[int(path[1:-1])-1][0])
        z_counter = z_counter + 1
    freeMocapTensor = tf.stack(subject_data_list)
    freeMocapTensor = normalize(freeMocapTensor)
    # print(freeMocapTensor[0][10].numpy())
    # print(newtensor.numpy())
    # print("Number of axes:", freeMocapTensor.ndim)
    print("Shape of Freemocap Tensor:", freeMocapTensor.shape)
    return freeMocapTensor


# ---------------------------------------------------SMI-TENSOR---------------------------------------------------------

# Returns the maximum number of seconds recorded from SMI glasses
def get_max_smi_seconds():
    max_seconds = 0
    for path in os.listdir(BUILDER_DIR):

        row = np.genfromtxt(os.path.join(BUILDER_DIR, os.path.join(path, 'BeGazeData.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1)
        # print(row.shape[0]) debug
        if row.shape[0] > max_seconds:
            max_seconds = row.shape[0]

    return max_seconds

# Creates a tensor for the SMI subjects
def SMITensor():
    num_subjects = len(subjects)
    max_seconds = get_max_smi_seconds()
    smiTensor = [[0] * max_seconds] * num_subjects
    z_counter = 0

    for path in os.listdir(BUILDER_DIR):
        # Code to Generate GSR Matrices of same time lengths
        row = np.genfromtxt(os.path.join(BUILDER_DIR, os.path.join(path, 'BeGazeData.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1, missing_values='').tolist()
        # print(path)
        tensor_index = int(path[1:-1]) - 1
        y_counter = len(row)
        while y_counter < max_seconds:
            row.append([0] * len(row[0]))
            y_counter += 1
        smiTensor[tensor_index] = row

        z_counter += 1

    smiTensor = tf.stack(smiTensor)
    print("Shape of SMI Tensor", smiTensor.shape)
    return smiTensor


# ---------------------------------------------NORMALIZATION_FUNCTIONS--------------------------------------------------

# Normalizes data without regards to a specific axis
def normalize(A):
    A = np.nan_to_num(A)
    Amax = np.max(A)
    # print(Amax)
    Amin = np.min(A)
    # print(Amin)
    range = Amax - Amin
    # print(range)
    Anrm = ((A - Amin) / range) * 2
    return Anrm

# Normalizes data in regards to the y-axis
def normalizeAxis(A):
    A = np.nan_to_num(A)
    Amax = np.max(A, axis=0, keepdims=True)
    # print(Amax)
    Amin = np.min(A, axis=0, keepdims=True)
    # print(Amin)
    range = Amax - Amin
    # print(Amax[0][:][0])
    # print(range[0][:][0])
    Anrm = ((A - Amin) / (range[0][:][0])) * 2
    return Anrm


# ------------------------------------------------iMOTIONS-TENSORS------------------------------------------------------

# Function which returns the number of instructors based on the number of folders in the Instructor directory
def get_num_instructors():
    num_instructors = 0  # initialized variable to hold the number of subject information
    instructors = []  # list to contain instructor names
    # Establishes number of subjects by checking how many files are in the subjects subdirectory and extract their names
    for path in os.listdir(INSTRUCTOR_DIR):
        instructors.append(path)  # Adds name of subject to subjects list
        num_instructors += 1

    # print('Number of Instructors:', num_instructors)
    # print('Instructor Names: ', instructors)
    subjects.extend(instructors)
    return num_instructors

# Returns the maximum number of seconds from each modality recording of the iMotions dataset
def get_max_instructor_seconds():
    max_seconds = 0
    for path in os.listdir(INSTRUCTOR_DIR):

        # Checks if eyegaze has the most number of seconds
        row = np.genfromtxt(os.path.join(INSTRUCTOR_DIR, os.path.join(path, 'EyeGazeAnalysis.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1)
        # print(row.shape[0]) debug
        if row.shape[0] > max_seconds:
            max_seconds = row.shape[0]

        # Checks if facial analysis has the most number of seconds
        row = np.genfromtxt(os.path.join(INSTRUCTOR_DIR, os.path.join(path, 'FacialAnalysis.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1)
        # print(row.shape[0]) debug
        if row.shape[0] > max_seconds:
            max_seconds = row.shape[0]

        # Checks if shimmer has the most number of seconds
        row = np.genfromtxt(os.path.join(INSTRUCTOR_DIR, os.path.join(path, 'ShimmerProcessed.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1)
        # print(row.shape[0]) debug
        if row.shape[0] > max_seconds:
            max_seconds = row.shape[0]

    return max_seconds

# Creates three tensors for each iMotions modality
def InstructorTensors():
    num_subjects = get_num_instructors()
    max_seconds = get_max_instructor_seconds()
    instructorTensors = [0] * 3
    # print(max_seconds) debug
    z_counter = 0
    eyeGazeData = [[0] * max_seconds] * num_subjects
    facialAnalysisData = [[0] * max_seconds] * num_subjects
    shimmerData = [[0] * max_seconds] * num_subjects

    for path in os.listdir(INSTRUCTOR_DIR):

        # Code to Generate Eye Gaze Matrices of same time lengths
        row = np.genfromtxt(os.path.join(INSTRUCTOR_DIR, os.path.join(path, 'EyeGazeAnalysis.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1, missing_values='').tolist()
        tensor_index = int(path[1:]) - 1
        y_counter = len(row)
        while y_counter < max_seconds:
            row.append([0] * len(row[0]))
            y_counter += 1
        eyeGazeData[tensor_index] = row

        # Code to Generate Facial Analysis Matrices of same time lengths
        row = np.genfromtxt(os.path.join(INSTRUCTOR_DIR, os.path.join(path, 'FacialAnalysis.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1, missing_values='').tolist()
        y_counter = len(row)
        while y_counter < max_seconds:
            row.append([0] * len(row[0]))
            y_counter += 1
        facialAnalysisData[tensor_index] = row

        # print(path)
        # print(len(row[0]))

        # Code to Generate GSR Matrices of same time lengths
        row = np.genfromtxt(os.path.join(INSTRUCTOR_DIR, os.path.join(path, 'ShimmerProcessed.csv')), delimiter=',',
                            dtype='float',
                            skip_header=1, missing_values='').tolist()
        y_counter = len(row)
        while y_counter < max_seconds:
            row.append(0)
            y_counter += 1
        shimmerData[tensor_index] = row

        z_counter += 1

    # Stacks the arrays into tensor objects
    eyeGazeTensor = tf.stack(eyeGazeData)
    eyeGazeTensor = normalizeAxis(eyeGazeTensor)
    facialAnalysisTensor = tf.stack(facialAnalysisData)
    facialAnalysisTensor = normalizeAxis(facialAnalysisTensor)
    shimmerTensor = tf.stack(shimmerData)
    shimmerTensor = normalize(shimmerTensor)

    print("Shape of Eyegaze Tensor:", eyeGazeTensor.shape)
    print("Shape of Facial Analysis Tensor:", facialAnalysisTensor.shape)
    print("Shape of Shimmer Tensor:", shimmerTensor.shape)

    instructorTensors[0] = eyeGazeTensor
    instructorTensors[1] = facialAnalysisTensor
    instructorTensors[2] = shimmerTensor

    return instructorTensors


# -----------------------------------------------TIME-TO-SECONDS--------------------------------------------------------

# Converts MM:SS time to number of seconds
def Time_Convert(x):
    times = x.split(':')
    return 60 * int(times[0]) + int(times[1])

# Converts timestamps in TIMESTAMP_CSV to seconds in Times.csv
def TimeToSeconds():
    df = pd.read_csv(TIMESTAMP_CSV)
    df['Start Task 1'] = df['Start Task 1'].apply(Time_Convert)
    df['End Task 1'] = df['End Task 1'].apply(Time_Convert)
    df['Start Task 2'] = df['Start Task 2'].apply(Time_Convert)
    df['End Task 2'] = df['End Task 2'].apply(Time_Convert)

    # print(df) debug

    df.to_csv('Times.csv', index=False)


# -------------------------------------------------MAKES-LABELS---------------------------------------------------------

# Uses the confuseometer outputs to make a list of tuples where (timestamp (in seconds), confusion level (0-3))
def MakeLabels():
    df = pd.read_csv(LABELS_CSV)
    labels = [0] * df.shape[0]
    rows = df.values
    i = 0
    for row in rows:
        labels[i] = (row[1][1:-1].split(', '))
        i += 1

    labels_list = []
    i = 0
    for label in labels:
        j = 0
        timestamp_list = []
        while j < len(label):
            first = int(int(label[j][1:]) / 10)
            second = int(label[j + 1][:-1])
            # print(first, second)
            # print("(", int(int(label[j][1:]) / 10), ", " + label[j + 1], sep='')
            label_tuple = (first, second)
            # print(label_tuple)
            timestamp_list.append(label_tuple)
            j += 2

        labels_list.append(timestamp_list)
        labels_list[i].insert(0, (0, 0))
        # print(i) debug
        # print(labels_list[i]) debug
        i += 1

    return labels_list


# Makes the labels into a tensor with appropriate times and indices
def LabelTensor():
    max_seconds = GetMaxSeconds()
    subjects_rating_list = [[0]] * len(subjects)
    tuples_list = MakeLabels()
    j = 0
    times = np.genfromtxt(os.path.join(TIMES_CSV), delimiter=',', skip_header=1, dtype=int)

    for subject in subjects:
        trial_num = 0
        index_num = 0

        if subject[0] == 'I':
            index_num = (int(subject[1:]) - 1) + int(len(subjects) / 2)
        elif subject[0] == 'B':
            index_num = int(subject[1:-1]) - 1

        if subject[0] == 'I':
            trial_num = (int(subject[1:]) - 1)
            # print(tuple_num)
        elif subject[0] == 'B':
            trial_num = int(subject[1:-1]) - 1

        # print(subject)
        tuples = tuples_list[index_num]
        rating_list = []
        tuple_num = 0
        i = 0
        while tuple_num < len(tuples) - 1:
            while i < tuples[tuple_num + 1][0]:
                rating_list += [(tuples[tuple_num][1])]
                i += 1
            tuple_num += 1
            # print(i)

        # print(len(rating_list))
        # print(times[trial_num][4])

        while i < times[trial_num][4]:
            rating_list += [(tuples[tuple_num][1])]
            i += 1
        # print(len(rating_list))

        rating_list = rating_list[times[trial_num][1]:times[trial_num][2]] + rating_list[
                                                                             times[trial_num][3]:times[trial_num][4]]

        while len(rating_list) < max_seconds:
            rating_list += [0]
        rating_list = to_categorical(rating_list, 4)

        # print(len(rating_list))
        # print(rating_list)

        # print(subject)
        # print(index_num)
        # print(times[trial_num][1])
        subjects_rating_list[index_num] = rating_list
        # print(subjects_rating_list[j])
        j += 1

    # print(subjects_rating_list[12])
    label_tensor = tf.stack(subjects_rating_list)
    label_tensor = label_tensor[:, 0:GetMinSeconds()]
    # print(label_tensor[15][125])
    print("Shape of Label Tensor", label_tensor.shape)
    return label_tensor


# -------------------------------------FINAL-TENSOR-CONSTRUCTION--------------------------------------------------------

# Gets the maximum amount of seconds that a trial took
def GetMaxSeconds():
    times = np.genfromtxt(os.path.join(TIMES_CSV), delimiter=',', skip_header=1, dtype=int)
    max_time = 0
    for row in times:
        current = (row[2] - row[1]) + (row[4] - row[3])
        if current > max_time:
            max_time = current

    return max_time


# Gets the minimum amount of seconds that a trial took
def GetMinSeconds():
    times = np.genfromtxt(os.path.join(TIMES_CSV), delimiter=',', skip_header=1, dtype=int)
    min_time = GetMaxSeconds()
    for row in times:
        current = (row[2] - row[1]) + (row[4] - row[3])
        if current < min_time:
            min_time = current

    return min_time


# Creates Final tensor with free mocap for the builders, and the iMotions data for the instructors
def FinalTensor(freeMocapTensor, smiTensor, instructorTensors):
    TimeToSeconds()  # Converts timestamps on one csv to seconds on another

    # Build tensor outline of size subjects x max_seconds x features
    max_seconds = GetMaxSeconds()
    final_tensor = [[0]] * len(subjects)
    # print(len(subjects))
    # print(max_seconds)
    # print(freeMocapTensor.shape[2] + 1)  # add instructor tensor sizes

    z_counter = 0
    for subject in subjects:
        subject_rows = [[0]] * max_seconds
        tensor_index = 0
        # print(subject_rows)

        if subject[0] == "B":
            # print("Builder")
            # print(subject)
            tensor_index = int(subject[1:-1]) - 1
            row = np.genfromtxt(os.path.join(TIMES_CSV), delimiter=',', skip_header=1, dtype=int)
            times = row[
                int(subject[1:-1]) - 1]  # array holding [index, start task 1, end task 1, start task 2, end task 2]
            # print(subject)
            # print(times[1])
            # print(freeMocapTensor[int(subject[1:-1]) - 1][times[1]].numpy())

            # Build tensor for task 1
            start = times[1]
            end = times[2] - start
            second = start
            adjustedSecond = 0
            while second <= end:
                # print(second)
                subject_rows[adjustedSecond] = [0]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               freeMocapTensor[int(subject[1:-1]) - 1][second].tolist()
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               smiTensor[int(subject[1:-1]) - 1][second].numpy().tolist()

                # Adds columns of zeros for iMotions data which is not applicable to the builder
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[0].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[1].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0]
                # print(len(subject_rows[second]))
                # subject_rows[second].extend([0] * 6)
                second += 1
                adjustedSecond += 1

            # print(subject_rows[0][:2])
            # print(instructorTensors[0].shape[2])
            # print(instructorTensors[1].shape[2])

            # Continue Building tensor for task 2
            start2 = times[3]
            end = times[4]
            second = start2
            while second <= end:
                subject_rows[adjustedSecond] = [0]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               freeMocapTensor[int(subject[1:-1]) - 1][second].tolist()
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               smiTensor[int(subject[1:-1]) - 1][second].numpy().tolist()
                # Adds columns of zeros for iMotions data which is not applicable to the builder
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[0].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[1].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0]
                # print(len(subject_rows[second]))
                # subject_rows[second].extend([0] * 6)
                second += 1
                adjustedSecond += 1

            # Fill in the rest of the rows with zeros to fit the tensor size
            while adjustedSecond < max_seconds:
                # print(freeMocapTensor.shape[2])
                subject_rows[adjustedSecond] = [0]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * freeMocapTensor.shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * smiTensor.shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[0].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[1].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0]
                adjustedSecond += 1
        elif subject[0] == "I":
            # print("Instructor")
            # print(subject[1:])
            tensor_index = (int(subject[1:]) - 1) + len(os.listdir(BUILDER_DIR))
            row = np.genfromtxt(os.path.join(TIMES_CSV), delimiter=',', skip_header=1, dtype=int)
            times = row[
                int(subject[1:]) - 1]  # array holding [index, start task 1, end task 1, start task 2, end task 2]
            # print(subject)
            # print(tensor_index)
            # print(times[1])
            # print(freeMocapTensor[int(subject[1:-1]) - 1][times[1]].numpy())

            # Build tensor for task 1
            start = times[1]
            end = times[2] - start
            second = start
            adjustedSecond = 0
            while second <= end:
                # print(second)

                subject_rows[adjustedSecond] = [1]

                # Adds columns of zeros for freeMocap data which is not applicable to the instructor
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * freeMocapTensor.shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * smiTensor.shape[2]

                # Adds columns for iMotions data
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               instructorTensors[0][int(subject[1:]) - 1][second].tolist()
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               instructorTensors[1][int(subject[1:]) - 1][second].tolist()
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [instructorTensors[2]
                                                                               [int(subject[1:]) - 1][second]]
                # print(len(subject_rows[second]))
                second += 1
                adjustedSecond += 1

            # Continue Building tensor for task 2
            start2 = times[3]
            end = times[4]
            second = start2
            while second <= end:
                subject_rows[adjustedSecond] = [1]

                # Adds columns of zeros for freeMocap data which is not applicable to the instructor
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * freeMocapTensor.shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * smiTensor.shape[2]

                # Adds columns for iMotions data
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               instructorTensors[0][int(subject[1:]) - 1][second].tolist()
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + \
                                               instructorTensors[1][int(subject[1:]) - 1][second].tolist()
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [instructorTensors[2]
                                                                               [int(subject[1:]) - 1][second]]
                # print(len(subject_rows[second]))
                # subject_rows[second].extend([0] * 6)
                second += 1
                adjustedSecond += 1

            # Fill in the rest of the rows with zeros to fit the tensor size
            while adjustedSecond < max_seconds:
                # print(freeMocapTensor.shape[2])
                subject_rows[adjustedSecond] = [1]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * freeMocapTensor.shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * smiTensor.shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[0].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0] * instructorTensors[1].shape[2]
                subject_rows[adjustedSecond] = subject_rows[adjustedSecond] + [0]
                adjustedSecond += 1

        # print(len(subject_rows[0]))
        # print(z_counter)
        final_tensor[tensor_index] = subject_rows
        z_counter += 1

    # print(final_tensor[15][0])
    tensor_object = tf.stack(final_tensor)
    # print(GetMinSeconds())
    tensor_object = tensor_object[:, 0:GetMinSeconds(), :]
    print("Shape of Final Tensor: ", tensor_object.shape)
    # print(tensor_object[1].numpy())
    return tensor_object


# -------------------------------------------------WINDOWING------------------------------------------------------------

# Creates windows of data according to the WINDOWSIZE timeframe
def windowing(finalTensor, labelTensor):
    # This creates converts finalTensor and labelTensor to size (160, WINDOWSIZE, 1642) and (160, WINDOWSIZE, 4)
    # respectively.
    # (No of Windows x timestamps in each window x length of features/labels )
    stepSize = 15
    windows = []
    labelWindows = []
    for i in range(finalTensor.shape[0]):
        for j in range(0, finalTensor.shape[1] - WINDOWSIZE + 1, stepSize):
            curr_window = []
            currLabelWindow = []
            for k in range(WINDOWSIZE):
                curr_window.append(finalTensor[i][j + k])
                currLabelWindow.append(labelTensor[i][j + k])
            if len(curr_window) == WINDOWSIZE:
                windows.append(curr_window)
                labelWindows.append(currLabelWindow)
    windows = tf.stack(windows)
    labelWindows = tf.stack(labelWindows)
    return windows, labelWindows


# -------------------------------------------------MAIN-METHOD----------------------------------------------------------

# Main function
def main():
    freeMocapTensor = FreeMocapTensor()
    smiTensor = SMITensor()
    # print(freeMocapTensor[2].numpy())
    instructorTensors = InstructorTensors()
    # transcribeTensor = TranscribeTensor()
    # acousticTensor = AcousticTensor()
    labelTensor = LabelTensor()
    # print("Names of All Subjects: ", subjects)
    finalTensor = FinalTensor(freeMocapTensor, smiTensor, instructorTensors)
    Run_RNN(finalTensor, labelTensor)


main()
