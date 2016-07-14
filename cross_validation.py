import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve


def cm2accuracy(cm):
    """
    counts accuracy by a confusion matrix
    """
    acc = float(sum(cm[i, i] for i in range(cm.shape[0])))
    return acc / cm.sum()


def learning_curves(X, y, clf, params, train_sizes=None, feature_selection=False, n_folds=3, scoring='accuracy'):
    """
    builds learning curves on test set, with parametesr chosen on train and validation set using nested cross validation
    :param X: data
    :param y: labels
    :param clf: classificator
    :param params: parameters for grid search
    :param train_sizes: train sizes for building learning curves
    :param feature_selection: whether to choose features by randomized logistic regression
    :param n_folds: number of outed cv folds
    :param scoring: scoring metric
    :return: train and test curve
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.5, 1.0, 5)

    kf = KFold(X.shape[0], n_folds=n_folds)

    train_curve = np.zeros_like(train_sizes)
    test_curve = np.zeros_like(train_sizes)

    for train_inds, test_inds in kf:
        train_data = X[train_inds]
        test_data = X[test_inds]
        train_labels = y[train_inds]
        test_labels = y[test_inds]

        if feature_selection:
            rlr = RandomizedLogisticRegression()
            rlr.fit(train_data, train_labels)

            inds = [i for i in range(X.shape[1]) if rlr.all_scores_[i] > 0.0]
            print len(inds)
            train_data = train_data[:, inds]

        gs = GridSearchCV(clf, params, scoring=scoring, cv=5)
        gs.fit(train_data, train_labels)
        bp = gs.best_params_
        print 'chosen params: ', bp

        for p in bp:
            setattr(clf, p, bp[p])
        lc = learning_curve(clf, test_data, test_labels, scoring=scoring, train_sizes=train_sizes)
        train_curve += lc[1].mean(axis=1)
        test_curve += lc[2].mean(axis=1)

    train_curve /= n_folds
    test_curve /= n_folds
    return train_curve, test_curve


def kfoldCM(classifier, data, labels, n_classes=None, n_folds=3):
    """
    performs cross-validation on a given dataset with a given classifier
    :param classifier: classifier
    :param data: dataset
    :param labels: labels for the dataset
    :param n_classes: number of classe (if none, determined automatically)
    :param n_folds: number of folds
    :return: confusion matrix for the whole dataset
    """
    kf = KFold(data.shape[0], n_folds=n_folds)
    classes = set(labels)
    if n_classes is None:
        n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))

    for train_inds, test_inds in kf:
        train_data = data[train_inds, :]
        train_labels = labels[train_inds]
        test_data = data[test_inds, :]
        test_labels = labels[test_inds]

        classifier.fit(train_data, train_labels)
        preds = classifier.predict(test_data)
        cm += confusion_matrix(test_labels, preds, labels=list(classes))
    return cm