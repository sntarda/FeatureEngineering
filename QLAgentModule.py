#_____REQUIRED LIBRARIES_____
from typing import Any
import cv2
import gensim
import random
import logging
import Augmentor
from google.colab import data_table
import numpy as np
import pandas as pd
from scipy import sparse
from skimage import exposure
import statsmodels.api as sm
from itertools import chain, combinations

#_____TF & KERAS & Torch_____
import torch
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Flatten, MaxPooling2D, MaxPool2D, BatchNormalization, Dropout, Bidirectional, SimpleRNN, Embedding, LSTM

#_____ML MODELS_____
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier  
from transformers import BertTokenizer, TFBertModel
from sklearn.ensemble import RandomForestClassifier
from transformers import TFBertForSequenceClassification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression

#_____DATA PROCESSING & FEATURE ENGINEERING_____
# For image augmentation
from PIL import Image
from skimage import exposure
from sklearn.preprocessing import LabelEncoder
# For Log Transformation
from numpy import log1p
# For Natural language processing
from gensim.models import Word2Vec, FastText
# For Image processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# For Text processing (e.g., n_gram)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# For preprocessing and feature selection
from sklearn.manifold import TSNE 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD, FastICA, PCA, NMF, KernelPCA
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import Normalizer, OneHotEncoder, KBinsDiscretizer, StandardScaler, MinMaxScaler, PolynomialFeatures, FunctionTransformer

# ____________________PROBLEM-TYPE & MODEL SELECTOR____________________ 

class MODEL_SELECTOR:
    def __init__(self):
        self.problem_type = None

    def select_problem_type(self):
        print("Select a problem type from the following options:")
        print("1. Regression")
        print("2. Binary_Classification")
        print("3. Multiclass_Classification")
        print("4. Clustering")
        print("5. Recommendation_System")
        print("6. Topic_Modeling")
        print("7. Image_Classification")
        print("8. Text_Classification")

        choice = input("Enter your choice (type the name of the problem): ")
        return choice.strip()

    def select_model(self, problem_type):
        algorithm_mapping = {
            "Regression": ["linear_rm", "random_forest_classifier", "SVM", "decision_tree_classifier", "knn_classifier"],
            "Binary_Classification": ["logistic_rm", "random_forest_classifier", "SVM", "Hist_Gradient_Boosting", "decision_tree_classifier", "knn_classifier", "basic_CNN", "advanced_CNN"],
            "Multiclass_Classification": ["random_forest_classifier", "SVM", "decision_tree_classifier", "SVM", "Hist_Gradient_Boosting", "knn_classifier", "basic_CNN", "advanced_CNN", "LDA"],
            "Image_Classification": ["basic_CNN", "advanced_CNN"],  # Basic and Advanced CNNs are typical for image classification
            "Text_Classification": ["logistic_rm", "random_forest_classifier", "SVM", "basic_RNN", "LSTM", "BERT"],  # Including various text classification algorithms
            "Clustering": ["K_menas"],
            "Recommendation_System": ["knn_classifier"],
            "Topic_Modeling": ["LDA"]
        }

        algorithms = algorithm_mapping.get(problem_type)
        if not algorithms:
            raise ValueError(f"No algorithms defined for: {problem_type}")
        return algorithms
    
# ____________________MODEL DESIGN____________________

class MODEL_BUILDER:
    def __init__(self, selected_algorithm):
        self.selected_algorithm = selected_algorithm
        self.model = None

    #___MODEL SELECTION & DESIGN___
    def build_model(self, kernel=None, n_clusters=None, n_neighbors=None):
        self.kernel = kernel
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

        #_____IMPORTANT CHECKS__________
        if self.selected_algorithm == 'SVM' and self.kernel is None:
            raise ValueError("You must specify 'kernel' for the SVM algorithm e.g., 'leaner', 'poly', 'rbf', 'sigmoid' .")
        if self.selected_algorithm == 'knn_classifier' and self.n_neighbors is None:
            raise ValueError("You must specify 'n_neighbors' for the SVM algorithm e.g., 2, 3, 4...")
        if self.selected_algorithm == 'knn_Regressor' and self.n_neighbors is None:
            raise ValueError("You must specify 'n_neighbors' for the SVM algorithm e.g., 2, 3, 4...")
        if self.selected_algorithm == 'K_means' and self.n_clusters is None:
            raise ValueError("You must specify 'n_clusters' for the KMeans algorithm e.g., 2, 3, 4...")

        algorithm_model_mapping = {
            "linear_rm": LinearRegression(),
            "logistic_rm": LogisticRegression(),
            "SVM": SVC(kernel=self.kernel) if self.kernel is not None else SVC(),
            "random_forest_classifier": RandomForestClassifier(),
            "decision_tree_classifier": DecisionTreeClassifier(),
            "decision_tree_regressor": DecisionTreeRegressor(),
            "Hist_Gradient_Boosting": HistGradientBoostingClassifier(),
            "knn_classifier": KNeighborsClassifier(n_neighbors=self.n_neighbors) if self.n_neighbors is not None else KNeighborsClassifier(),
            "knn_regressor": KNeighborsRegressor(n_neighbors=self.n_neighbors) if self.n_neighbors is not None else KNeighborsRegressor(),
            "K_menas": KMeans(n_clusters=self.n_clusters) if self.n_clusters is not None else KMeans(),
            "LDA": LinearDiscriminantAnalysis(),
            "basic_CNN": self.basic_CNN_model(),
            "advanced_CNN": self.advanced_CNN_model(),
            "LSTM": self.LSTM_model(),
            "BERT": self.BERT_model(),
            #"basic_RNN": self.basic_RNN_model(),#________ To be designed
            #"advanced_RNN": self.basic_RNN_model()#______ To be designed
        }

        self.model = algorithm_model_mapping.get(self.selected_algorithm)
        if not self.model:
            raise ValueError(f"No model defined for: {self.selected_algorithm}")
        
        return self.model

    def basic_CNN_model(self):
        model = Sequential()
        model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))  
        model.add(MaxPool2D())
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def advanced_CNN_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def LSTM_model(self):
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(60, 4)))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dense(units=1, activation='softmax'))  # or 'sigmoid' for binary classification
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # adjust loss and metrics as needed
        return model

    def BERT_model(self):
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')  # Load pre-trained model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Further code to fine-tune the model on your specific dataset
        return model, tokenizer

    #___MODELS OPTIMIZATION___
    def optimize_model(self, train_data, train_labels, param_grid, optimization_method='grid', n_iter=10):
        """
        Optimizes the model using either GridSearchCV or RandomizedSearchCV.      
        Parameters:
        - param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        - optimization_method: The method of optimization ('grid' for GridSearchCV or 'random' for RandomizedSearchCV).
        - n_iter: Number of parameter settings that are sampled in RandomizedSearchCV. Defaults to 10.
        Returns:
        - The best model after performing optimization.
        """
        self.train_data = train_data
        self.train_labels = train_labels

        if not self.model:
            raise ValueError("No model has been set for optimization.")
        if optimization_method == 'grid':
            optimizer = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        elif optimization_method == 'random':
            optimizer = RandomizedSearchCV(self.model, param_grid, n_iter=n_iter, cv=5, scoring='accuracy', random_state=42)
        else:
            raise ValueError("Invalid optimization method. Choose 'grid' or 'random'.")
        # Fitting the optimizer to the data
        optimizer.fit(self.train_data, self.train_labels)
        # Updating the model to the best found model
        self.model = optimizer.best_estimator_

        return self.model

# ____________________FEATURE ENGINEERING METHODS DESIGN____________________

class DATA_ENGINEERING:

    METHODS_ACTIONS = {
        "standard_scaler": 'standard_scaling',
        "MinMax": 'MinMaxScaler',
        "pca": 'principal_component_analysis',  
        "rfe": 'recursive_feature_elimination',    
        "lesso_selector": 'lasso_regularization_selector',
        "anova": 'anova_f_test_selector',
        "chi_selector": 'chi_squared_selector',
        "mutual_selector": 'mutual_info_selector',
        "log_transform": 'log_transformation',
        "polynomial": 'polynomial_features',
        "variance_threshold": 'variance_threshold_selector',
        "kernel_pca": 'kernel_pca',
        "ica": 'independent_component_analysis',
        "nmf": 'non_negative_matrix_factorization',
        "svd": 'singular_value_decomposition',
        "agglomeration": 'feature_agglomeration',
        "k_binning": 'binning',
        "tsne": 'tsne_embedding',
        "one_hot_encoding": 'one_hot_encoding',
        "tfidf": 'tfidf_vectorization',
        "count_vectorizer": 'count_vectorization',
        "histogram_equalizer": 'histogram_equalization',
        "image_resizing": 'image_resizing',
        "image_standardizer": 'images_standardization',
        "image_augmenter": 'image_augmentation',
        "image_normalizer": 'image_normalization',
        "label_encoder": 'encode_labels',
        "word_embedding": 'word2vec_embedding',
        "fast_embedding": 'fasttext_embedding'
    }

    def __init__(self):
        self.data_transformer = None
    
    def MinMaxScaler(self, data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)

    def standard_scaling(self, data): 
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def principal_component_analysis(self, data, n_components=1):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)
    
    def recursive_feature_elimination(self, data, labels):
        estimator = LogisticRegression()
        n_features_to_select = 5
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        return selector.fit_transform(data, labels)
    
    def lasso_regularization_selector(self, data, labels):
        lasso = LassoCV()
        lasso.fit(data, labels)
        return data[:, lasso.coef_ != 0]

    def anova_f_test_selector(self, data, labels, k=10):
        anova_selector = SelectKBest(f_classif, k=k)
        return anova_selector.fit_transform(data, labels)

    def chi_squared_selector(self, data, labels, k=10):
        chi2_selector = SelectKBest(chi2, k=k)
        return chi2_selector.fit_transform(data, labels)
    
    def mutual_info_selector(self, data, labels, k=10):
        mi_selector = SelectKBest(mutual_info_classif, k=k)
        return mi_selector.fit_transform(data, labels)
    
    def log_transformation(self, data):
        # Ensure data does not contain negative values
        if np.any(data <= 0):
            data = np.where(data <= 0, np.nan, data)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data = imputer.fit_transform(data)
        return np.log1p(data)
    
    def polynomial_features(self, data, degree=2):
        poly = PolynomialFeatures(degree=degree)
        return poly.fit_transform(data)

    def variance_threshold_selector(self, data, threshold=0.0):
        selector = VarianceThreshold(threshold)
        return selector.fit_transform(data)

    def kernel_pca(self, data, n_components=1, kernel='rbf'):
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        return kpca.fit_transform(data)

    def independent_component_analysis(self, data, n_components=1):
        ica = FastICA(n_components=n_components)
        return ica.fit_transform(data)

    def non_negative_matrix_factorization(self, data, n_components=1, tol=1e-4, max_iter=200):
        if np.any(data < 0):
            data = np.where(data < 0, np.nan, data)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data = imputer.fit_transform(data)
        try:
            model = NMF(n_components=n_components, tol=tol, max_iter=max_iter)
            features = model.fit_transform(data)
        except ValueError as e:
            features = data
        return features

    def singular_value_decomposition(self, data, n_components=1):
        svd = TruncatedSVD(n_components=n_components)
        return svd.fit_transform(data)
    
    def feature_agglomeration(self, data, n_clusters=2):
        agglomeration = FeatureAgglomeration(n_clusters=n_clusters)
        return agglomeration.fit_transform(data)

    def binning(self, data, n_bins=5):
        if np.isnan(data).any():
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data = imputer.fit_transform(data)
        try:
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
            binned_data = discretizer.fit_transform(data)
        except Exception as e:
            print(f"Binning failed: {e}")
            binned_data = data
        return binned_data
    
    def tsne_embedding(self, data, n_components=1, perplexity=30.0):
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return tsne.fit_transform(data)
    
    def one_hot_encoding(self, data):
        encoder = OneHotEncoder(handle_unknown='ignore')
        return encoder.fit_transform(data).toarray()
    
    def tfidf_vectorization(self, text_data):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(text_data) 
    
    def count_vectorization(self, data):
        n = 2  # for bigrams; adjust for different n-grams
        n_grams = [' '.join(grams) for text in data for grams in zip(*[text[i:] for i in range(n)])]
        count_vectorizer = CountVectorizer()
        return count_vectorizer.fit_transform(n_grams)
    
    def histogram_equalization(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(img_gray)
        return exposure.equalize_hist(equalized_img)
    
    def image_resizing(self, data, target_size=(224, 224)):
        resized_images = []
        for image_path in data:
            image = Image.open(image_path)
            image = image.resize(target_size)
            resized_images.append(image)
        return resized_images

    def image_augmentation(self, data):
        # Assuming data is a list of file paths to the images
        p = Augmentor.Pipeline()

        # Define your augmentation operations
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.crop_random(probability=0.7, percentage_area=0.8)
        p.resize(probability=1.0, width=120, height=120)

        # Now we apply the operations to the images
        augmented_images = []
        for image_path in data:
            image = Image.open(image_path)
            # Augmentor works by adding operations to a pipeline and then executing them
            # Here, we use a temporary pipeline for each image
            temp_pipeline = Augmentor.Pipeline()
            temp_pipeline.set_seed(100)  # Optional: for reproducibility
            temp_pipeline.add_operation(p.operations)
            augmented_image = temp_pipeline.sample_with_array(image)
            augmented_images.append(augmented_image)
        return augmented_images
    
    def image_normalization(self, data):
        normalized_images = []
        for image in data:
            norm_image = np.array(image) / 255.0  # Normalizing pixel values
            normalized_images.append(norm_image)
        return normalized_images
    
    def images_standardization(self, data):
        standardized_images = []
        for image in data:
            std_image = (image - np.mean(image)) / np.std(image)
            standardized_images.append(std_image)
        return standardized_images
    
    def encode_labels(self, labels):
        encoder = LabelEncoder()
        return encoder.fit_transform(labels)

    def word2vec_embedding(self, data):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model_w2v = Word2Vec(data, vector_size=100, window=5, min_count=1, workers=4)
        model_w2v.save("word2vec.model")
        return model_w2v

    def fasttext_embedding(self, data):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model_ft = FastText(vector_size=100, window=5, min_count=1, workers=4, word_ngrams=1)
        model_ft.build_vocab(sentences=data)
        model_ft.train(sentences=data, total_examples=len(data), epochs=model_ft.epochs)
        model_ft.save("fasttext.model")
        return model_ft

    def engineer_data(self, data, method, transform_only=False, **kwargs):
        if method not in DATA_ENGINEERING.METHODS_ACTIONS:
            raise ValueError(f"No method defined for: {method}")

        # Get the method object from the method name
        method_obj = getattr(self, DATA_ENGINEERING.METHODS_ACTIONS[method])
        
        # If only transform is needed and the method has a 'transform' attribute
        if transform_only and hasattr(method_obj, 'transform'):
            return method_obj.transform(data)
        else:
            return method_obj(data, **kwargs)
           
# ____________________FEATURE METHODS DICTIONARY (classified by Algorithm use)____________________

METHODS_REQUIRE_LABELS = ['rfe', 'lesso_selector', 'anova', 'chi_selector', 'mutual_selector']

ALGORITHM_METHODS = {
'linear_rm': {
  'variance_threshold': DATA_ENGINEERING.variance_threshold_selector,
  'anova_selector': DATA_ENGINEERING.anova_f_test_selector,
  'lesso_selector': DATA_ENGINEERING.lasso_regularization_selector,
  'pca': DATA_ENGINEERING.principal_component_analysis,
  'kernel_pca': DATA_ENGINEERING.kernel_pca,
  'ica': DATA_ENGINEERING.independent_component_analysis,
  'nmf': DATA_ENGINEERING.non_negative_matrix_factorization,
  'svd': DATA_ENGINEERING.singular_value_decomposition,
  'agglomeration': DATA_ENGINEERING.feature_agglomeration,
  'standard_scaler': DATA_ENGINEERING.standard_scaling,
  'MinMax': DATA_ENGINEERING.MinMaxScaler,
  'log_transform': DATA_ENGINEERING.log_transformation,
  'polynomial': DATA_ENGINEERING.polynomial_features
},
'decision_tree_classifier': {
           "standard_scaler": DATA_ENGINEERING.standard_scaling,
           "pca": DATA_ENGINEERING.principal_component_analysis,
           "log_transform": DATA_ENGINEERING.log_transformation,
           "variance_threshold": DATA_ENGINEERING.variance_threshold_selector,
           "kernel_pca": DATA_ENGINEERING.kernel_pca,
           "ica": DATA_ENGINEERING.independent_component_analysis,
           "nmf": DATA_ENGINEERING.non_negative_matrix_factorization,
           "svd": DATA_ENGINEERING.singular_value_decomposition,
           "agglomeration": DATA_ENGINEERING.feature_agglomeration,
           "k_binning": DATA_ENGINEERING.binning
},
'Hist_Gradient_Boosting': {
           "standard_scaler": DATA_ENGINEERING.standard_scaling,
           "pca": DATA_ENGINEERING.principal_component_analysis,
           "log_transform": DATA_ENGINEERING.log_transformation,
           "variance_threshold": DATA_ENGINEERING.variance_threshold_selector,
           "kernel_pca": DATA_ENGINEERING.kernel_pca,
           "ica": DATA_ENGINEERING.independent_component_analysis,
           "nmf": DATA_ENGINEERING.non_negative_matrix_factorization,
           "svd": DATA_ENGINEERING.singular_value_decomposition,
           "agglomeration": DATA_ENGINEERING.feature_agglomeration,
           "k_binning": DATA_ENGINEERING.binning
},
'knn_classifier': {
           "standard_scaler": DATA_ENGINEERING.standard_scaling,
           "pca": DATA_ENGINEERING.principal_component_analysis,
           "log_transform": DATA_ENGINEERING.log_transformation,
           "variance_threshold": DATA_ENGINEERING.variance_threshold_selector,
           "kernel_pca": DATA_ENGINEERING.kernel_pca,
           "ica": DATA_ENGINEERING.independent_component_analysis,
           "nmf": DATA_ENGINEERING.non_negative_matrix_factorization,
           "svd": DATA_ENGINEERING.singular_value_decomposition,
           "agglomeration": DATA_ENGINEERING.feature_agglomeration,
           "k_binning": DATA_ENGINEERING.binning
},
'basic_CNN': {
           "standard_scaler": DATA_ENGINEERING.standard_scaling,
           "image_resizing": DATA_ENGINEERING.image_resizing,
           "image_augmenter": DATA_ENGINEERING.image_augmentation,
           "image_normalizer": DATA_ENGINEERING.image_normalization,
           "image_standardizer": DATA_ENGINEERING.images_standardization,
           "label_encoder": DATA_ENGINEERING.encode_labels
},
'advanced_CNN': {
           "standard_scaler": DATA_ENGINEERING.standard_scaling,
           "image_resizing": DATA_ENGINEERING.image_resizing,
           "image_augmenter": DATA_ENGINEERING.image_augmentation,
           "image_normalizer": DATA_ENGINEERING.image_normalization,
           "image_standardizer": DATA_ENGINEERING.images_standardization,
           "label_encoder": DATA_ENGINEERING.encode_labels
}}

# ____________________FEATURE METHODS COMBINATION GENERATOR____________________

class COMBO_GENERATOR:
    def __init__(self, algorithm_methods):
        # Ensure algorithm_methods is a list of strings
        if not all(isinstance(method, str) for method in algorithm_methods):
            raise TypeError("algorithm_methods should be a list of strings")
        self.algorithm_methods = algorithm_methods
    def generate_combos(self):
        # Generate combinations:
        all_combinations = []
        # Update the length (decide the number of feature methods in each combination)
        for i in range(1, 4):
            all_combinations.extend(combinations(self.algorithm_methods, i))
        # Randomly sample combinations (generate all possible random combinations or limit it to a speicific count)
        selected_combinations = random.sample(all_combinations, min(len(all_combinations), 10)) if all_combinations else []
        # Print selected combinations
        for comb in selected_combinations:
            print(", ".join(comb))
        return selected_combinations
    
# ____________________MODULE END_________________________________
# add additional customization (classes, functions, methods, etc)