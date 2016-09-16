import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier as rf_class
from sknn.mlp import Classifier, Layer


def make_predictions(train, train_labels, test, predictor):
    if predictor == 'forest':
        ## Train classifier ##
        clf = rf_class(n_estimators=100)
        clf.fit(train.T, train_labels)
        ## Predict ##
        pred = clf.predict(test)
    elif predictor == 'neural_net':
        # Try neural net
        nn = Classifier(
            layers=[
                Layer("Rectifier", units=100),
                Layer("Softmax")],
            learning_rate=0.02,
            n_iter=10)
        nn.fit(np.array(train.T), np.array(train_labels))
        pred = nn.predict(np.array(test))

    return pred
