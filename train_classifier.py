from glob import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from util import paths_to_images_gen
from features import extract_features_many
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.image as mpimg
import pickle

RAND_STATE = 10
MODEL_FILE = 'svc.pkl'

# Use Small Dataset
CARS_GLOB = './data/vehicles_smallset/*/*.jpeg'
NO_CARS_GLOB = './data/non-vehicles_smallset/*/*.jpeg'

# Use Full Dataset
# CARS_GLOB = './data/vehicles/*/*.png'
# NO_CARS_GLOB = './data/non-vehicles/*/*.png'

# dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
# svc = dist_pickle["svc"]
# X_scaler = dist_pickle["scaler"]
# orient = dist_pickle["orient"]
# pix_per_cell = dist_pickle["pix_per_cell"]
# cell_per_block = dist_pickle["cell_per_block"]
# spatial_size = dist_pickle["spatial_size"]
# hist_bins = dist_pickle["hist_bins"]

MODEL_KEY = 'model'
SCALER_KEY = 'scaler'


def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    clf = SVC(kernel='rbf', C=5)
    # clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Trained model with", score, "accuracy")

    store = {
        MODEL_KEY: clf,
        SCALER_KEY: scaler
    }
    with open(MODEL_FILE, 'wb') as model_f:
        pickle.dump(store, model_f)


def load_model(test_accuracy=False):
    with open(MODEL_FILE, 'rb') as model_f:
        stored = pickle.load(model_f)
    if not test_accuracy:
        return stored

    svc = stored[MODEL_KEY]
    scaler = stored[SCALER_KEY]
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y, scaler)
    score = svc.score(X_test, y_test)
    print("Loaded model with", score, "accuracy")
    return stored


def tune_svc_params():
    """ Find the best parameters to use on our support vector machine """
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    svc = SVC()
    parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 5, 10, 20]}
    grid_search = GridSearchCV(svc, parameters)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)


def tune_rfc_params():
    """ Find the best parameters to use on our support vector machine """
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    svc = RandomForestClassifier()
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 5, 10, 20]}
    # grid_search = GridSearchCV(svc, parameters)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)


def prepare_data(X, y, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    scaled_X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=RAND_STATE)
    return X_train, X_test, y_train, y_test, scaler


def load_data():
    car_paths = glob(CARS_GLOB)
    no_car_paths = glob(NO_CARS_GLOB)

    show_img_size(car_paths[0])

    # Detect image paths and load images
    car_imgs = paths_to_images_gen(car_paths)
    car_features = extract_features_many(car_imgs)
    no_car_imgs = paths_to_images_gen(no_car_paths)
    no_car_features = extract_features_many(no_car_imgs)
    y = np.concatenate((np.ones(len(car_paths)), np.zeros(len(no_car_paths))))
    X = np.vstack((car_features, no_car_features)).astype(np.float64)
    return X, y


def show_img_size(path):
    img = mpimg.imread(path)
    print("Training data image size:", img.shape)
    pass


if __name__ == '__main__':
    train_model()
    load_model()
