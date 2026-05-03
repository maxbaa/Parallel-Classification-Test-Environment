__author__ = "Satoshi Kashima"

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from multiprocessing import Pool

class Argument:
    def __init__(self, X, y, n_estimators, num_samples, num_features, num_max_features, num_important_features,
                 ranked_features_indices, important_features_indices, max_depth):
        self.X = X
        self.y = y
        self.n_estimators = n_estimators
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_max_features = num_max_features
        self.num_important_features = num_important_features
        self.ranked_features_indices = ranked_features_indices
        self.important_features_indices = important_features_indices
        self.max_depth = max_depth

    def get(self):
        return (self.X, self.y, self.n_estimators, self.num_samples, self.num_features, self.num_max_features,
                self.num_important_features, self.ranked_features_indices, self.important_features_indices,
                self.max_depth)

class Output:
    def __init__(self, estimators, oobs):
        self.estimators = estimators
        self.oobs = oobs

    def get(self):
        return self.estimators, self.oobs




def compute_variable_importance(x: np.ndarray, y: np.ndarray, parent_entropy: float):
    best_information_gain = 0
    # best_information_ratio = float('-inf')
    unique_values = np.unique(x)
    for value in unique_values:
        left_mask = x <= value

        # ensure that there is no split such that there is no samples at one of the children
        if np.sum(left_mask) == 0 or np.sum(left_mask) == len(left_mask):
            continue

        right_mask = ~left_mask
        left_labels = y[left_mask]
        right_labels = y[right_mask]
        entropy = (compute_entropy(left_labels) * len(left_labels) + compute_entropy(right_labels) * len(
            right_labels)) / len(y)
        information_gain = parent_entropy - entropy
        # information_ratio = information_gain / compute_entropy(x)

        # if information_ratio > best_information_ratio:
        #     best_information_ratio = information_ratio

        if information_gain > best_information_gain:
            best_information_gain = information_gain

    # return best_information_ratio
    return best_information_gain


def compute_entropy(labels):
    class_counts = np.bincount(labels)
    num_samples = np.sum(class_counts)
    probabilities = class_counts / num_samples
    return -np.sum(probabilities * np.log2(probabilities + 1e-12))  # 0 gives an error


def create_decision_trees(arg):
    X, y, n_estimators, num_samples, num_features, num_max_features, num_important_features, ranked_features_indices, important_features_indices, max_depth = arg.get()
    estimators = []
    oobs = []

    # actual procedure
    for _ in range(n_estimators):
        # Bagging
        sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)

        # need to select features (important features + randomly selected features)
        # and samples
        if num_features > 5:
            randomly_chosen_indices = np.random.randint(num_important_features, num_features,
                                                        size=num_max_features - num_important_features)
            selected_feature_indices = np.append(important_features_indices,
                                                 ranked_features_indices[randomly_chosen_indices])
            X_subset = X[np.ix_(sample_indices, selected_feature_indices)]
        else:
            selected_feature_indices = None  # not applicable
            X_subset = X[sample_indices]

        y_subset = y[sample_indices]

        # create a decision tree and fit it to the subset
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X_subset, y_subset)

        # add the decision tree to the ensemble
        estimators.append([tree, selected_feature_indices])
        oobs.append(sample_indices)

    return Output(estimators, oobs)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=5, random_state=None, n_processes=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.num_max_features = None  # number of max features used in each tree
        self.estimators = []
        self.num_important_features = None  # the k value explained in the paper
        self.n_processes = n_processes

        self.weights = np.ones(n_estimators)  # weight of each tree estimator
        self.features_selected = False

    def fit(self, X, y):
        num_samples, num_features = X.shape
        np.random.seed(self.random_state)
        self.estimators = []
        oobs = []
        ranked_features_indices = None
        important_features_indices = None

        # set num_max_features and num_important_variables if they are None
        if num_features > 5 and self.num_max_features is None and self.num_important_features is None:
            self.num_max_features = int(np.ceil(np.sqrt(num_features)))
            self.num_important_features = int(np.ceil((3 / 5) * self.num_max_features))
        elif num_features <= 5 and self.num_max_features is None:
            self.num_max_features = num_features
        else:
            # todo add error handling
            pass

        # dimension reduction - selecting top k important variables
        if num_features > 5:
            self.features_selected = True
            parent_entropy = compute_entropy(y)

            # try splitting for each feature
            variable_importances = np.zeros((num_features, 2), dtype=int)
            for i in range(num_features):
                vi = compute_variable_importance(X[:, i], y, parent_entropy)
                variable_importances[i, :] = i, vi

            ranked_features_indices = variable_importances[variable_importances[:, 1].argsort()[::-1], 0]
            important_features_indices = ranked_features_indices[:self.num_important_features]

        n_processes = max(1, self.n_processes)
        estimators_per_process = [self.n_estimators // n_processes] * n_processes
        for i in range(self.n_estimators % n_processes):
            estimators_per_process[i] += 1

        args = [
            (
                Argument(
                    X,
                    y,
                    n_estimators_for_process,
                    num_samples,
                    num_features,
                    self.num_max_features,
                    self.num_important_features,
                    ranked_features_indices,
                    important_features_indices,
                    self.max_depth,
                ),
            )
            for n_estimators_for_process in estimators_per_process
            if n_estimators_for_process > 0
        ]

        if n_processes == 1:
            results = [create_decision_trees(*arg) for arg in args]
        else:
            try:
                pool = Pool(processes=n_processes)
                async_results = [pool.apply_async(create_decision_trees, args=arg) for arg in args]
                pool.close()
                pool.join()
                results = [res.get() for res in async_results]
            except (OSError, PermissionError):
                results = [create_decision_trees(*arg) for arg in args]

        for res in results:
            _estimators, _oobs = res.get()
            self.estimators.extend(_estimators)
            oobs.extend(_oobs)


        # setting the weight of the model using OOB data
        for i, estimator in enumerate(self.estimators):
            tree = estimator[0]
            feature_indices = estimator[1]
            oob_indices = oobs[i]
            sample_mask = np.bincount(oob_indices, minlength=num_samples) == 0

            X_oob = X[sample_mask]
            y_oob = y[sample_mask]

            if self.features_selected:
                X_oob = X_oob[:, feature_indices]

            if len(y_oob) == 0:
                self.weights[i] = 0.0
                continue

            predictions = tree.predict(X_oob)
            weight = np.sum(predictions == y_oob) / len(y_oob)
            self.weights[i] = weight

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.estimators)))

        for i, (tree, feature_indices) in enumerate(self.estimators):
            X_subset = X if feature_indices is None else X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)

        return np.apply_along_axis(get_max_class, axis=1, arr=predictions, weights=self.weights)

def get_max_class(predictions, weights):
    hashmap = {}
    for pred, weight in zip(predictions, weights):
        if pred not in hashmap:
            hashmap[pred] = 0

        hashmap[pred] += weight

    max_class = None
    max_weigh_total = 0
    for key, val in hashmap.items():
        if val > max_weigh_total:
            max_weigh_total = val
            max_class = key

    return max_class
