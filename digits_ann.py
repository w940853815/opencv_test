import gzip
import pickle
import cv2
import numpy as np


def load_data():
    mnist = gzip.open("data/mnist.pkl.gz", "rb")
    training_data, test_data = pickle.load(mnist)
    mnist.close()
    return (training_data, test_data)


def wrap_data():
    tr_d, te_d = load_data()
    training_inputs = tr_d[0]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_data = zip(te_d[0], te_d[1])
    return (training_data, test_data)


def vectorized_result(j):
    e = np.zeros((10,), np.float32)
    e[j] = 1.0
    return e


def create_ann(hidden_nodes=60):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([784, hidden_nodes, 10]))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 1.0))
    return ann


def train(ann, samples=50000, epochs=10):
    tr, test = wrap_data()
    tr = list(tr)
    for epoch in range(epochs):
        print(f"Completed {epoch}/{epochs} epochs")
        counter = 0
        for img in tr:
            if counter > samples:
                break
            if counter % 100 == 0:
                print(f"Epoch {epoch}: Trained on {counter}/{samples} samples")
            counter += 1
            sample, response = img
            data = cv2.ml.TrainData_create(
                np.array([sample], dtype=np.float32),
                cv2.ml.ROW_SAMPLE,
                np.array([response], dtype=np.float32),
            )
            if ann.isTrained():
                ann.train(
                    data,
                    cv2.ml.ANN_MLP_UPDATE_WEIGHTS
                    | cv2.ml.ANN_MLP_NO_INPUT_SCALE
                    | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE,
                )
            else:
                ann.train(
                    data, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE
                )
    print("Completed all epochs!")
    return ann, test


def predict(ann, sample):
    if sample.shape != (784,):
        if sample.shape != (28, 28):
            sample = cv2.resize(sample, (28, 28), interpolation=cv2.INTER_LINEAR)
        sample = sample.reshape(
            784,
        )
    return ann.predict(np.array([sample], np.float32))


def test(ann, test_data):
    num_tests = 0
    num_correct = 0
    for img in test_data:
        num_tests += 1
        sample, correct_digit_class = img
        digit_class = predict(ann, sample)[0]
        if digit_class == correct_digit_class:
            num_correct += 1
    print(f"Accuracy: {(100.0 * num_correct /num_tests)}%")


ann, test_data = train(create_ann())
test(ann, test_data)
