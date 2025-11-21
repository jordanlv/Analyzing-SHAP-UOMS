import numpy as np
import os
import argparse
import pickle
import copy
import gc

import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.gmm import GMM


def train_and_get_results(clf, X, random_state):

    X_train, X_test = train_test_split(X, train_size=0.8, random_state=random_state)

    scl = StandardScaler().fit(X_train)
    X_train = scl.transform(X_train)
    X_test = scl.transform(X_test)

    clf.fit(X_train)

    shp = shap.KernelExplainer(
        clf.decision_function,
        shap.kmeans(X_train, min(50, X_train.shape[0])),
    )

    return {
        "predictions": clf.predict(X_test),
        "scores": clf.decision_function(X_test),
        "shapvalues": shp(X_test).values,
    }


def compute_experiment(
    dataset_path,
    save_path,
    classifiers,
    n_fold=5,
    seed=0,
):
    print(dataset_path)

    # Load Dataset
    data = np.load(dataset_path, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Save ground truth
    for i in range(n_fold):
        y_train, y_test = train_test_split(y, train_size=0.8, random_state=i)
        os.makedirs(f"{save_path}/ground_truth", exist_ok=True)
        with open(f"{save_path}/ground_truth/{i}.pkl", "wb") as f:
            pickle.dump(y_test, f)

    for clf_name, base_clf in classifiers.items():

        clf = copy.deepcopy(base_clf)

        print(clf_name)

        os.makedirs(f"{save_path}/{clf_name}", exist_ok=True)

        setattr(clf, "n_features", X.shape[1])
        setattr(clf, "random_state", seed)
        setattr(clf, "verbose", 0)

        for i in range(n_fold):

            with open(
                f"{save_path}/{clf_name}/{i}.pkl",
                "wb",
            ) as f:
                pickle.dump(train_and_get_results(clf, X, i), f)

        gc.collect()


# Parse arguments
parser = argparse.ArgumentParser(description="Process dataset ID.")
parser.add_argument("--id", type=int, required=True, help="ID of the dataset")
args = parser.parse_args()
dataset_id = args.id

# Set random seed
seed = 0
np.random.seed(0)

# All datasets we want to use
datasets = [
    "2_annthyroid",
    "4_breastw",
    "14_glass",
    "15_Hepatitis",
    "21_Lymphography",
    "23_mammography",
    "27_PageBlocks",
    "29_Pima",
    "37_Stamps",
    "38_thyroid",
    "39_vertebral",
    "40_vowels",
    "42_WBC",
    "44_Wilt",
    "45_wine",
    "47_yeast",
]

classifiers = {
    "AutoEncoder": AutoEncoder(),
    "CBLOF": CBLOF(),
    "HBOS": HBOS(),
    "IForest": IForest(),
    "KNN": KNN(),
    "LOF": LOF(),
    "MCD": MCD(),
    "OCSVM": OCSVM(),
    "PCA": PCA(),
    "ECOD": ECOD(),
    "COPOD": COPOD(),
    "LODA": LODA(),
    "DeepSVDD": DeepSVDD(n_features=1),
    "GMM": GMM(),
}


# Define dataset path
dataset_path = f"datasets/{datasets[dataset_id]}.npz"

# Define save path
save_dir = "results"

os.makedirs(save_dir, exist_ok=True)

save_path = f"{save_dir}/{datasets[dataset_id]}"
os.makedirs(save_path, exist_ok=True)


compute_experiment(dataset_path, save_path, classifiers, n_fold=5, seed=seed)
