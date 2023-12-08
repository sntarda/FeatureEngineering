# IMPORT NECESSARY LIBRARIES:
import cv2
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec, FastText
import logging 
import Augmentor
from scipy import sparse
from skimage import exposure
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, KBinsDiscretizer, Normalizer, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import statsmodels.api as sm
from gensim.models import Word2Vec as word2vec_embedding
from gensim.models import FastText as fasttext_embedding
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE as tsne_embedding
from sklearn.decomposition import KernelPCA, NMF, LatentDirichletAllocation, FastICA, TruncatedSVD
from sklearn.decomposition import FastICA as independent_component_analysis
from sklearn.decomposition import KernelPCA as kernel_pca
from sklearn.decomposition import NMF as non_negative_matrix_factorization
from sklearn.decomposition import KernelPCA as kernel_pca, FastICA as independent_component_analysis
from sklearn.decomposition import TruncatedSVD as singular_value_decomposition
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf_vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import Normalizer as normalization
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer as KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, Binarizer, FunctionTransformer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
# Traditional Machine Learning Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import GeneralizedLinearRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # or DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier  # or KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# MODEL SELECTOR 
class MLModelSelector:
    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.model = None

    def algorithm_selector(self):
        selected_algorithm = None

        if self.problem_type == "Prediction":
            selected_algorithm = "linear-rm"
        elif self.problem_type == "Prediction2":
            selected_algorithm = "SVM"
        elif self.problem_type == "Prediction(count type)":
            selected_algorithm = "generalized-lm"
        elif self.problem_type == "Prediction(true/fales)":
            selected_algorithm = "logistic_rm"
        elif self.problem_type == "Prediction(multi-outcome)":
            selected_algorithm = "decision_tree"
        elif self.problem_type == "Clustering":
            selected_algorithm = "K-menas"
        elif self.problem_type == "Text Classification":
            selected_algorithm = "RNN"
        elif self.problem_type == "Image Classification":
            selected_algorithm = "CNN"
        elif self.problem_type == "Topic Modeling":
            selected_algorithm = "LDA"
        elif self.problem_type == "Opinion Mining":
            selected_algorithm = "RNN"
        elif self.problem_type == "Recommend Systems":
            selected_algorithm = "KNN"
        else:
            raise ValueError(f"No algorithm defined for: {self.problem_type}")

        return selected_algorithm

# ALGORITHM FEATURE METHODS
# PolynomialFeatures in sklearn is used for creating polynomial features and interactions.
polynomial_interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# For interactions only, we can use the same PolynomialFeatures but with a specific parameter.
interaction_only = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# FEATURE ENGINEERING METHODS:

def variance_threshold_selector(data, threshold=0.05):
    if sparse.issparse(data):
        data = data.toarray()  
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(data)

def chi_squared_selector(data, target):
    k_value = 10
    selector = SelectKBest(score_func=chi2, k=k_value)
    return selector.fit_transform(data, target)

def anova_f_test_selector(data, target):
    k_value = 10
    selector = SelectKBest(score_func=f_classif, k=k_value)
    return selector.fit_transform(data, target)

def mutual_info_selector(data, target):
    k_value = 10
    selector = SelectKBest(score_func=mutual_info_classif, k=k_value)
    return selector.fit_transform(data, target)

def recursive_feature_elimination(data, target):
    estimator = LogisticRegression()
    n_features_to_select = 5
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    return selector.fit_transform(data, target)

def lasso_regularization_selector(data, target):
    alpha_value = 0.1
    lasso = Lasso(alpha=alpha_value)
    lasso.fit(data, target)
    return data[:, lasso.coef_ != 0]

def principal_component_analysis(data, n_components=2):
    if sparse.issparse(data):
        data = data.toarray()
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def linear_discriminant_analysis(data, target):
    n_components_value = 2
    lda = LinearDiscriminantAnalysis(n_components=n_components_value)
    return lda.fit_transform(data, target)

def kernel_pca(data):
    n_components_value = 2
    kernel_pca = KernelPCA(n_components=n_components_value, kernel="rbf")
    return kernel_pca.fit_transform(data)

def independent_component_analysis(data, n_components=2):
    if sparse.issparse(data):
        data = data.toarray()  
    try:
        ica = FastICA(n_components=n_components)
        return ica.fit_transform(data)
    except ValueError as e:
        print(f"ICA failed: {e}. Skipping...")
        return data

def non_negative_matrix_factorization(data, n_components=2):

    # Check for negative values
    has_negative = np.any(data < 0)
    if has_negative:
        data -= np.min(data)

    try:
        model = NMF(n_components=n_components)  
        features = model.fit_transform(data)
    except ValueError as e:
        features = None
        
    if features is None:    
        features = data

    return features

def tsne_embedding(data):
    model = TSNE(n_components=2)
    return model.fit_transform(data)

def singular_value_decomposition(data):
    svd = TruncatedSVD(n_components=2)
    return svd.fit_transform(data)

def feature_agglomeration(data):
    agglo = FeatureAgglomeration(n_clusters=2)
    return agglo.fit_transform(data)

def standard_scaling(data):
    if sparse.issparse(data):
        data = data.toarray()  
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def min_max_scaling(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def one_hot_encoding(data):

    if sparse.issparse(data):
        data = data.toarray() 

    try:
        encoder = OneHotEncoder(handle_unknown='ignore') 
        encoded_data = encoder.fit_transform(data)
    except Exception as e:
        print(f"One-hot encoding failed: {e}")
        encoded_data = data

    return encoded_data

def binning(data, n_bins=5):

    if sparse.issparse(data):
        data = data.toarray()

    try:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        binned_data = discretizer.fit_transform(data)
    except Exception as e:
        print(f"Binning failed: {e}")
        binned_data = data

    return binned_data

def tfidf_vectorization(data):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(data)

def count_vectorization(data):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data)

def normalization(data):
    normalizer = Normalizer()
    return normalizer.fit_transform(data)

def log_transformation(data):

    # Check for sparse input
    if sparse.issparse(data):
        data = data.toarray()

    # Handle negative values
    if np.any(data < 0):
        data = data + abs(data.min()) + 1

    try:
        transformed = np.log1p(data)
    except Exception as e:
        print(f"Log transform failed: {e}")
        transformed = data

    return transformed

def feature_scaling(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def polynomial_features(data, degree=2):
    poly = PolynomialFeatures(degree=degree, interaction_only=True)
    return poly.fit_transform(data)

def histogram_equalization(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(img_gray)
    return exposure.equalize_hist(equalized_img)

def image_augmentation(data):
    # Assuming the data is an image. This function can be more complex based on your specific augmentation needs.
    p = Augmentor.Pipeline()
    # Define your augmentation pipeline here...
    images = np.array([p.augment_image(img) for img in data])
    return images

def n_grams(data):
    n = 2  # for bigrams; adjust for different n-grams
    n_grams = [' '.join(grams) for text in data for grams in zip(*[text[i:] for i in range(n)])]
    count_vectorizer = CountVectorizer()
    return count_vectorizer.fit_transform(n_grams)

def word2vec_embedding(data):
    # Set up logging for Gensim, useful to get insights into the training process
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_w2v = Word2Vec(data, vector_size=100, window=5, min_count=1, workers=4)
    # Save the model to a file for later use
    model_w2v.save("word2vec.model")
    return model_w2v

def fasttext_embedding(data):
    """
    Train a FastText model based on the provided data.

    Parameters:
    data (iterable): Iterable of sentences. A sentence is represented by a list of strings (words).

    Returns:
    FastText: A trained FastText model.
    """
    # Set up logging for Gensim, useful to get insights into the training process
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Training the FastText model
    model_ft = FastText(vector_size=100, window=5, min_count=1, workers=4, word_ngrams=1)  # word_ngrams=1 means using the standard FastText training method
    model_ft.build_vocab(sentences=data)
    model_ft.train(sentences=data, total_examples=len(data), epochs=model_ft.epochs)  # train based on your data

    # Save the model to a file for later use
    model_ft.save("fasttext.model")

    return model_ft

METHODS_ACTIONS = {"variance_threshold_selector":variance_threshold_selector,
                    "chi_squared_selector":chi_squared_selector,
                    "anova_f_test_selector":anova_f_test_selector,
                    "mutual_info_selector":mutual_info_selector,
                    "recursive_feature_elimination":recursive_feature_elimination,
                    "lasso_regularization_selector":lasso_regularization_selector,
                    "principal_component_analysis":principal_component_analysis,
                    "linear_discriminant_analysis":linear_discriminant_analysis,
                    "kernel_pca":kernel_pca,
                    "independent_component_analysis":independent_component_analysis,
                    "non_negative_matrix_factorization":non_negative_matrix_factorization,
                    "tsne_embedding":tsne_embedding,
                    "singular_value_decomposition":singular_value_decomposition,
                    "feature_agglomeration":feature_agglomeration,
                    "StandardScaler":standard_scaling,
                    "MinMaxScaler":MinMaxScaler,
                    "OneHotEncoding":one_hot_encoding,
                    "Binning":binning,
                    "LogTransformation":log_transformation,
                    "PolynomialFeature-Interaction":polynomial_features,
                    "tfidf_vectorization":tfidf_vectorization,
                    "count_vectorization":count_vectorization,
                    "normalization":normalization,
                    "feature_scaling":feature_scaling,
                    "word2vec_embedding":word2vec_embedding,
                    "fasttext_embedding":fasttext_embedding,
                    "n_grams":n_grams,
                    "image_augmentation":image_augmentation,
                    "histogram_equalization":histogram_equalization}


# RUN MODEL
# Select the model based on the algorithm
algorithms_dict = {"linear_rm":LinearRegression,
                "polynomial_rm":PolynomialFeatures,
                "SVM": SVC,
                #"generalized-lm":GeneralizedLinearRegressor,
                "logistic_rm":LogisticRegression,
                "decision_tree":DecisionTreeClassifier,
                "KNN":KNeighborsClassifier,
                "K-means":KMeans,
                "RNN":LSTM,
                "CNN":Conv2D,
                "LDA":LinearDiscriminantAnalysis}

class RunModel:
    def __init__(self, selected_algorithm):
        self.selected_algorithm = selected_algorithm
        self.model = None

    def algorithm_processor(self, data):
        selected_algorithm = self.selected_algorithm

        if selected_algorithm == "linear_rm":
            self.model = LinearRegression()

        elif selected_algorithm == "SVM":
            # Create SVM classifier without training it here
            self.model = SVC(kernel='linear')

        elif self.selected_algorithm == "RFM":
            # Create Random Forest classifier
            self.model = RandomForestClassifier(n_estimators=100,  
                                                random_state=42,  
                                                max_depth=None, 
                                                min_samples_split=2,  
                                                min_samples_leaf=1)
            
        elif selected_algorithm == "generalized_lm":
            endog = data['y']  # Replace 'y' with the actual column name of the dependent variable in your dataset
            exog = sm.add_constant(data[['x']])
            self.model = sm.GLM(endog, exog, family=sm.families.Poisson())

        elif selected_algorithm == "logistic_rm":
            self.model = LogisticRegression()

        elif selected_algorithm == "decision_tree":
            self.model = DecisionTreeClassifier(random_state=42)

        elif selected_algorithm == "K_menas":
            self.model = KMeans()

        elif selected_algorithm == "CNN":
            # self.model = tf.keras.Sequential()
            # if selected_algorithm == "CNN":
            #    self.model.add(Conv2D(32, (3, 3), activation='relu'))  # just an example
            #    self.model.add(MaxPooling2D())
            #    self.model.add(Flatten())
            #    self.model.add(Dense(10, activation='softmax'))  # assuming 10 classes
            #self.model.compile(optimizer='adam',  # or another optimizer
            #                   loss='sparse_categorical_crossentropy',  # or another loss function
            #                   metrics=['accuracy'])  # or other metrics
            data = data.reshape(data.shape[0], data.shape[1], 1)
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(32, 1, activation='relu' , input_shape=(data.shape[1], 1)),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        elif selected_algorithm == "RNN":
            self.model = Sequential()
            # Example for a simple RNN. You should adjust the architecture to your needs.
            self.model.add(LSTM(32, input_shape=(input_length, input_dim)))  # adjust input_shape to your data
            self.model.add(Dense(10, activation='softmax'))  # adjust according to your problem
            self.model.compile(optimizer='adam', 
                               loss='categorical_crossentropy', 
                               metrics=['accuracy'])
            
        elif selected_algorithm == "LDA":
            self.model = LinearDiscriminantAnalysis()

        elif self.selected_algorithm == "KNN":
            self.model = KNeighborsClassifier()
        else:
            raise ValueError(f"No model defined for: {self.selected_algorithm}")

        return self.model
    
ALGORITHM_METHODS = {
'linear_rm': {
  'variance_threshold_selector': VarianceThreshold(),
  'anova_f_test_selector': SelectKBest(score_func=f_classif),
  'lasso_regularization_selector': Lasso(),
  'principal_component_analysis': PCA(),
  'kernel_pca': KernelPCA(),
  'independent_component_analysis': FastICA(),
  'non_negative_matrix_factorization': NMF(),
  'singular_value_decomposition': TruncatedSVD(),
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler(),
  'MinMaxScaler': MinMaxScaler(),
  'LogTransformation': FunctionTransformer(np.log1p),
  'PolynomialFeature-Interaction': PolynomialFeatures(include_bias=False),
  'Interaction-Only': PolynomialFeatures(interaction_only=True, include_bias=False),
  'SimpleImputer': SimpleImputer()
},
'SVM': {
  'StandardScaler': StandardScaler(),
  'MinMaxScaler': MinMaxScaler(),
  'LogTransformation': FunctionTransformer(np.log1p),
  #'PolynomialFeature-Interaction': PolynomialFeatures(include_bias=False, interaction_only=True),
  'VarianceThresholdSelector': VarianceThreshold(),
  #'ANOVA_F_Test_Selector': SelectKBest(score_func=f_classif),
  'LassoRegularizationSelector': Lasso(),
  'PrincipalComponentAnalysis': PCA(),
  'KernelPCA': KernelPCA(),
  'IndependentComponentAnalysis': FastICA(),
  'SingularValueDecomposition': TruncatedSVD(),
  'NonNegativeMatrixFactorization': NMF(),
  'FeatureAgglomeration': FeatureAgglomeration()
},
'RFM': {
  'StandardScaler': StandardScaler(), 
  'MinMaxScaler': MinMaxScaler(),  
  'LogTransformation': FunctionTransformer(np.log1p), 
  'PolynomialFeature-Interaction': PolynomialFeatures(include_bias=False, interaction_only=True),
  'VarianceThresholdSelector': VarianceThreshold(), 
  'ANOVA_F_Test_Selector': SelectKBest(score_func=f_classif),
  'PrincipalComponentAnalysis': PCA(), 
  'KernelPCA': KernelPCA(),  
  'IndependentComponentAnalysis': FastICA(),
  'SingularValueDecomposition': TruncatedSVD(), 
  'NonNegativeMatrixFactorization': NMF(),  
  'FeatureAgglomeration': FeatureAgglomeration() 
},
'polynomial_rm': {
  'variance_threshold_selector': VarianceThreshold(),
  'anova_f_test_selector': SelectKBest(score_func=f_classif),
  'lasso_regularization_selector': Lasso(),
  'principal_component_analysis': PCA(),
  'kernel_pca': KernelPCA(),
  'independent_component_analysis': FastICA(),
  'non_negative_matrix_factorization': NMF(),
  'singular_value_decomposition': TruncatedSVD(),
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler(),
  'MinMaxScaler': MinMaxScaler(),
  'LogTransformation': FunctionTransformer(np.log1p),
  'PolynomialFeature-Interaction': PolynomialFeatures(include_bias=False),
  'Interaction-Only': PolynomialFeatures(interaction_only=True, include_bias=False),
  'SimpleImputer': SimpleImputer()
},
'generalized_lm': {
  'variance_threshold_selector': VarianceThreshold(),
  'anova_f_test_selector': SelectKBest(score_func=f_classif),
  'lasso_regularization_selector': Lasso(),
  'principal_component_analysis': PCA(),
  'kernel_pca': KernelPCA(),
  'independent_component_analysis': FastICA(),
  'non_negative_matrix_factorization': NMF(),
  'singular_value_decomposition': TruncatedSVD(),
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler(),
  'MinMaxScaler': MinMaxScaler(),
  'LogTransformation': FunctionTransformer(np.log1p),
  'PolynomialFeature-Interaction': PolynomialFeatures(include_bias=False),
  'Interaction-Only': PolynomialFeatures(interaction_only=True, include_bias=False),
  'SimpleImputer': SimpleImputer()
},
'logistic_rm': {
  'variance_threshold_selector': VarianceThreshold(),
  'anova_f_test_selector': SelectKBest(score_func=f_classif),
  'lasso_regularization_selector': Lasso(),
  'principal_component_analysis': PCA(),
  'kernel_pca': KernelPCA(),
  'independent_component_analysis': FastICA(),
  'non_negative_matrix_factorization': NMF(),
  'singular_value_decomposition': TruncatedSVD(),
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler(),
  'MinMaxScaler': MinMaxScaler(),
  'LogTransformation': FunctionTransformer(np.log1p),
  'PolynomialFeature-Interaction': PolynomialFeatures(include_bias=False),
  'Interaction-Only': PolynomialFeatures(interaction_only=True, include_bias=False),
  'SimpleImputer': SimpleImputer()
},
'decision_tree': {
  'variance_threshold_selector': VarianceThreshold(),
  'principal_component_analysis': PCA(),
  'kernel_pca': KernelPCA(),
  'independent_component_analysis': FastICA(),
  'non_negative_matrix_factorization': NMF(),
  'singular_value_decomposition': TruncatedSVD,
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler,
  #'OneHotEncoding': OneHotEncoder(),  # Assuming OneHotEncoder is intended here
  #'DummyEncoding': dummy_encoding,  # This function needs the dataframe and columns as input
  'Binning': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),
  #'LogTransformation': log_transformer(),
  'PolynomialFeature-Interaction': polynomial_interaction,
  #'Interaction-Only': interaction_only,
  #'SimpleImputer': SimpleImputer
},
'KNN': {
  'variance_threshold_selector': VarianceThreshold(),
  'principal_component_analysis': PCA(),
  'kernel_pca': kernel_pca,
  'independent_component_analysis': independent_component_analysis,
  'non_negative_matrix_factorization': non_negative_matrix_factorization,
  'singular_value_decomposition': singular_value_decomposition,
  'feature_agglomeration': FeatureAgglomeration(),
  'standard_scaling': StandardScaler,
  #'OneHotEncoding': OneHotEncoder(),
  #'DummyEncoding': dummy_encoding,  # This requires a dataframe and specific columns as inputs
  'Binning': KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'),  # Example parameters
  'LogTransformation': log_transformation,
  #'PolynomialFeature-Interaction': PolynomialFeatures(degree=2, include_bias=False),
  #'Interaction_Only': PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
  'SimpleImputer': SimpleImputer(),
  'feature_scaling': MinMaxScaler()
}
,
'K-means': {
  'variance_threshold_selector': VarianceThreshold(),
  'principal_component_analysis': PCA(),
  'kernel_pca': kernel_pca,
  'independent_component_analysis': independent_component_analysis,
  'non_negative_matrix_factorization': non_negative_matrix_factorization,
  'tsne_embedding': tsne_embedding,
  'singular_value_decomposition': singular_value_decomposition,
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler,
  'MinMaxScaler': MinMaxScaler(),
  'SimpleImputer': SimpleImputer(),
},
'RNN': {
  'recursive_feature_elimination': RFE(estimator=SVR(kernel="linear"), n_features_to_select=1, step=1),
  'principal_component_analysis': PCA(),
  'kernel_pca': kernel_pca,
  'independent_component_analysis': independent_component_analysis,
  'non_negative_matrix_factorization': non_negative_matrix_factorization,
  'singular_value_decomposition': singular_value_decomposition,
  'feature_agglomeration': FeatureAgglomeration(),
  'StandardScaler': StandardScaler,
  'MinMaxScaler': MinMaxScaler(),
  'SimpleImputer': SimpleImputer(),
  'tfidf_vectorization': tfidf_vectorization,
  'count_vectorization': CountVectorizer(),
  'word2vec_embedding': word2vec_embedding,
  'fasttext_embedding': fasttext_embedding,
  'n_grams': CountVectorizer(ngram_range=(1, 2))
},
'CNN': {
  'recursive_feature_elimination': RFE(estimator=SVR(kernel="linear"), n_features_to_select=1, step=1),
  'principal_component_analysis': PCA(),
  'kernel_pca': kernel_pca,
  'independent_component_analysis': independent_component_analysis,
  'non_negative_matrix_factorization': non_negative_matrix_factorization,
  'singular_value_decomposition': singular_value_decomposition,
  'feature_agglomeration': FeatureAgglomeration(),
  'standard_scaling': StandardScaler(),
  'MinMaxScaler': MinMaxScaler(),
  'SimpleImputer': SimpleImputer(),
  'normalization': normalization,
  'image_augmentation': image_augmentation,
  'histogram_equalization': histogram_equalization
},
'LDA': {
  'recursive_feature_elimination': RFE(estimator=SVR(kernel="linear"), n_features_to_select=1, step=1),
  'principal_component_analysis': PCA(),
  'kernel_pca': kernel_pca,
  'independent_component_analysis': independent_component_analysis,
  'non_negative_matrix_factorization': non_negative_matrix_factorization,
  'singular_value_decomposition': singular_value_decomposition,
  'feature_agglomeration': FeatureAgglomeration(),
  'standard_scaling': StandardScaler,
  'MinMaxScaler': MinMaxScaler,
  'SimpleImputer': SimpleImputer,
  'tfidf_vectorization': tfidf_vectorization,
  'count_vectorization': CountVectorizer(),
  'word2vec_embedding': word2vec_embedding,
  'fasttext_embedding': fasttext_embedding,
  'n_grams': CountVectorizer(),
}


}
