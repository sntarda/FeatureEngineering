from google.colab import drive
drive.mount('/content/drive')

# UPLOAD CUSTOM MODULE & DATASETS
from google.colab import files
uploaded = files.upload()
!ls

# IMPORT MODULE/S:

import QLAgentModule
from QLAgentModule import MODEL_SELECTOR, MODEL_BUILDER, METHODS_REQUIRE_LABELS, ALGORITHM_METHODS, COMBO_GENERATOR, DATA_ENGINEERING
#METHODS_REQUIRE_LABELS = QLAgentModule.METHODS_REQUIRE_LABELS

import QLAgentEnvironment
from QLAgentEnvironment import FeatureEngineeringEnvironment, QLearningAgent, train_evaluate_agent, visualize_results

"""#**IRIS DATASET**


*   **Problem Type:** Multi-class Classification
*   **Model used:** Decision Tree Classifier


"""

# __________Iris Dataset______________
from sklearn.datasets import load_iris
DATA, LABELS = load_iris(return_X_y=True)

print(f"Iris Dataset:\n")
print(f"Data Shape: {DATA.shape}")
print(f"Labels Shape: {LABELS.shape}")
print(f"Data Shape: {DATA.dtype}")
print(f"Labels Shape: {LABELS.dtype}")
print("-"*50)

# __________MODULE CONNECTION_________
selector = MODEL_SELECTOR()
# Select the problem type
problem_type = selector.select_problem_type()
model_options = selector.select_model(problem_type)
print(f"Select a suitable model for {problem_type}: {model_options}")
# User inputs the selected model
SELECTED_MODEL = input("Enter your selected model from the list above: ")

GET_MODEL = MODEL_BUILDER(SELECTED_MODEL)
MODEL = GET_MODEL.build_model()
FEATURE_METHODS_DIC = ALGORITHM_METHODS[SELECTED_MODEL]
FEATURE_METHODS = list(FEATURE_METHODS_DIC.keys())
ALL_METHODS_COMBINATIONS = COMBO_GENERATOR(FEATURE_METHODS).generate_combos()
if not ALL_METHODS_COMBINATIONS:
    raise ValueError("No feature method combinations were generated.")

print(f"========================== MODEL & FIEATURES SUMMARY ==========================")
print(f"\nSELECTED MODEL:")
print(f"{MODEL}\n")
print("FEATURE_METHODS:")
for idx, method in enumerate(FEATURE_METHODS, start=1):
    print(f"{idx}. {method}")
print("\nFEATURE_METHODS COMBINATIONS:")
for idx, combo in enumerate(ALL_METHODS_COMBINATIONS, start=1):
    print(f"    {idx}. {combo}")

print("-"*50)

# __________REQUIRED PARAMETERS________
NUM_ACTIONS = len(ALL_METHODS_COMBINATIONS)
NUM_STATES = NUM_ACTIONS + 1
MAX_EPISODES = 10

# __________ENVIRONMENT INITIALIZATION________
ENV = FeatureEngineeringEnvironment(DATA, LABELS, MODEL, FEATURE_METHODS,
                                    ALL_METHODS_COMBINATIONS)

# __________QL-AGENT INITIALIZATION___________
AGENT = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS,
                        lr_min=0.001, lr_max=0.1, df=0.999, eps=1.0, eps_min=0.1,
                        eps_decay=0.001, decay=0.05, window_size=50,
                        total_episodes=MAX_EPISODES)

# __________GET INIIAL MODEL ACCURACY____________
# To set the initial accuracy.
ENV.get_initial_accuracy(DATA, LABELS, MODEL, num_folds=5)
print(f"Initial Accuracy: {ENV.initial_accuracy}")
print("-"*50)

#___________OUTPUTS & VISUALIZATIONS____________________

# Get outputs
OUTPUTS = train_evaluate_agent(ENV, AGENT, ALL_METHODS_COMBINATIONS,
                                MAX_EPISODES)
# Unpack outputs
episode_rewards, episode_accuracies, top_combinations, q_table = OUTPUTS

# Visualize the results of the Q-Learning agent.
VISUALIZE_OUTPUTS = visualize_results(episode_rewards, episode_accuracies,
                                      q_table, NUM_ACTIONS, top_combinations)

"""#**BREAST CANCER DATASET**

*   **Problem Type:** Multi-class Classification
*   **Model used:** Decision Tree Classifier
"""

# __________Breast Cancer Dataset______________
from sklearn.datasets import load_breast_cancer
DATA, LABELS = load_breast_cancer(return_X_y=True)

print(f"Breast Cancer Dataset:\n")
print(f"Data Shape: {DATA.shape}")
print(f"Labels Shape: {LABELS.shape}")
print(f"Data Shape: {DATA.dtype}")
print(f"Labels Shape: {LABELS.dtype}")
print("-"*50)

# __________MODULE CONNECTION_________
selector = MODEL_SELECTOR()
# Select the problem type
problem_type = selector.select_problem_type()
model_options = selector.select_model(problem_type)
print(f"Select a suitable model for {problem_type}: {model_options}")
# User inputs the selected model
SELECTED_MODEL = input("Enter your selected model from the list above: ")

GET_MODEL = MODEL_BUILDER(SELECTED_MODEL)
MODEL = GET_MODEL.build_model()
FEATURE_METHODS_DIC = ALGORITHM_METHODS[SELECTED_MODEL]
FEATURE_METHODS = list(FEATURE_METHODS_DIC.keys())
ALL_METHODS_COMBINATIONS = COMBO_GENERATOR(FEATURE_METHODS).generate_combos()
if not ALL_METHODS_COMBINATIONS:
    raise ValueError("No feature method combinations were generated.")

print(f"========================== MODEL & FIEATURES SUMMARY ==========================")
print(f"\nSELECTED MODEL:")
print(f"{MODEL}\n")
print("FEATURE_METHODS:")
for idx, method in enumerate(FEATURE_METHODS, start=1):
    print(f"{idx}. {method}")
print("\nFEATURE_METHODS COMBINATIONS:")
for idx, combo in enumerate(ALL_METHODS_COMBINATIONS, start=1):
    print(f"    {idx}. {combo}")

print("-"*50)

# __________REQUIRED PARAMETERS________
NUM_ACTIONS = len(ALL_METHODS_COMBINATIONS)
NUM_STATES = NUM_ACTIONS + 1
MAX_EPISODES = 10

# __________ENVIRONMENT INITIALIZATION________
ENV = FeatureEngineeringEnvironment(DATA, LABELS, MODEL, FEATURE_METHODS,
                                    ALL_METHODS_COMBINATIONS)

# __________QL-AGENT INITIALIZATION___________
AGENT = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS,
                        lr_min=0.001, lr_max=0.1, df=0.999, eps=1.0, eps_min=0.1,
                        eps_decay=0.001, decay=0.05, window_size=50,
                        total_episodes=MAX_EPISODES)

# __________GET INIIAL MODEL ACCURACY____________
# To set the initial accuracy.
ENV.get_initial_accuracy(DATA, LABELS, MODEL, num_folds=5)
print(f"Initial Accuracy: {ENV.initial_accuracy}")
print("-"*50)

#___________OUTPUTS & VISUALIZATIONS____________________

# Get outputs
OUTPUTS = train_evaluate_agent(ENV, AGENT, ALL_METHODS_COMBINATIONS,
                                MAX_EPISODES)
# Unpack outputs
episode_rewards, episode_accuracies, top_combinations, q_table = OUTPUTS

# Visualize the results of the Q-Learning agent.
VISUALIZE_OUTPUTS = visualize_results(episode_rewards, episode_accuracies,
                                      q_table, NUM_ACTIONS, top_combinations)

"""#**CALFIORNIA HOUSING DATASET**



*   **Problem Type:** Regression
*   **Model Used:** Linear Regression


"""

# __________CALFIORNIA HOUSING DATASET______________
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
DATA = housing.data
LABELS = housing.target

print(f"Breast Cancer Dataset:\n")
print(f"Data Shape: {DATA.shape}")
print(f"Labels Shape: {LABELS.shape}")
print(f"Data Shape: {DATA.dtype}")
print(f"Labels Shape: {LABELS.dtype}")
print("-"*50)

# __________MODULE CONNECTION_________
selector = MODEL_SELECTOR()
# Select the problem type
problem_type = selector.select_problem_type()
model_options = selector.select_model(problem_type)
print(f"Select a suitable model for {problem_type}: {model_options}")
# User inputs the selected model
SELECTED_MODEL = input("Enter your selected model from the list above: ")

GET_MODEL = MODEL_BUILDER(SELECTED_MODEL)
MODEL = GET_MODEL.build_model()
FEATURE_METHODS_DIC = ALGORITHM_METHODS[SELECTED_MODEL]
FEATURE_METHODS = list(FEATURE_METHODS_DIC.keys())
ALL_METHODS_COMBINATIONS = COMBO_GENERATOR(FEATURE_METHODS).generate_combos()
if not ALL_METHODS_COMBINATIONS:
    raise ValueError("No feature method combinations were generated.")

print(f"========================== MODEL & FIEATURES SUMMARY ==========================")
print(f"\nSELECTED MODEL:")
print(f"{MODEL}\n")
print("FEATURE_METHODS:")
for idx, method in enumerate(FEATURE_METHODS, start=1):
    print(f"{idx}. {method}")
print("\nFEATURE_METHODS COMBINATIONS:")
for idx, combo in enumerate(ALL_METHODS_COMBINATIONS, start=1):
    print(f"    {idx}. {combo}")

print("-"*50)

# __________REQUIRED PARAMETERS________
NUM_ACTIONS = len(ALL_METHODS_COMBINATIONS)
NUM_STATES = NUM_ACTIONS + 1
MAX_EPISODES = 10

# __________ENVIRONMENT INITIALIZATION________
ENV = FeatureEngineeringEnvironment(DATA, LABELS, MODEL, FEATURE_METHODS,
                                    ALL_METHODS_COMBINATIONS)

# __________QL-AGENT INITIALIZATION___________
AGENT = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS,
                        lr_min=0.001, lr_max=0.1, df=0.999, eps=1.0, eps_min=0.1,
                        eps_decay=0.001, decay=0.05, window_size=50,
                        total_episodes=MAX_EPISODES)

# __________GET INIIAL MODEL ACCURACY____________
# To set the initial accuracy.
ENV.get_initial_accuracy(DATA, LABELS, MODEL, num_folds=5)
print(f"Initial Accuracy: {ENV.initial_accuracy}")
print("-"*50)

#___________OUTPUTS & VISUALIZATIONS____________________

# Get outputs
OUTPUTS = train_evaluate_agent(ENV, AGENT, ALL_METHODS_COMBINATIONS,
                                MAX_EPISODES)
# Unpack outputs
episode_rewards, episode_accuracies, top_combinations, q_table = OUTPUTS

# Visualize the results of the Q-Learning agent.
VISUALIZE_OUTPUTS = visualize_results(episode_rewards, episode_accuracies,
                                      q_table, NUM_ACTIONS, top_combinations)

"""#**DIABETES DATASET**


*   **Problem Type:** Regression
*   **Model Used:** Linear Regression


"""

# ______________________DIABETES DATASET________________________________________
from sklearn.datasets import load_diabetes
import pandas as pd

# Load the dataset
diabetes_dataset = load_diabetes()
data = pd.DataFrame(diabetes_dataset.data, columns=diabetes_dataset.feature_names)
data['progression'] = diabetes_dataset.target

DATA = data.drop('progression', axis=1)
LABELS = data['progression']

print(f"Train Data Shape: {DATA.shape}")
print(f"Train Label Shape: {LABELS.shape}")
print(f"Test Data Shape: {DATA.shape}")
print(f"Test Label Shape: {LABELS.shape}")
print("-"*50)

# __________MODULE CONNECTION_________
selector = MODEL_SELECTOR()
# Select the problem type
problem_type = selector.select_problem_type()
model_options = selector.select_model(problem_type)
print(f"Select a suitable model for {problem_type}: {model_options}")
# User inputs the selected model
SELECTED_MODEL = input("Enter your selected model from the list above: ")

GET_MODEL = MODEL_BUILDER(SELECTED_MODEL)
MODEL = GET_MODEL.build_model()
FEATURE_METHODS_DIC = ALGORITHM_METHODS[SELECTED_MODEL]
FEATURE_METHODS = list(FEATURE_METHODS_DIC.keys())
ALL_METHODS_COMBINATIONS = COMBO_GENERATOR(FEATURE_METHODS).generate_combos()
if not ALL_METHODS_COMBINATIONS:
    raise ValueError("No feature method combinations were generated.")

print(f"========================== MODEL & FIEATURES SUMMARY ==========================")
print(f"\nSELECTED MODEL:")
print(f"{MODEL}\n")
print("FEATURE_METHODS:")
for idx, method in enumerate(FEATURE_METHODS, start=1):
    print(f"{idx}. {method}")
print("\nFEATURE_METHODS COMBINATIONS:")
for idx, combo in enumerate(ALL_METHODS_COMBINATIONS, start=1):
    print(f"    {idx}. {combo}")

print("-"*50)

# __________REQUIRED PARAMETERS________
NUM_ACTIONS = len(ALL_METHODS_COMBINATIONS)
NUM_STATES = NUM_ACTIONS + 1
MAX_EPISODES = 10

# __________ENVIRONMENT INITIALIZATION________
ENV = FeatureEngineeringEnvironment(DATA, LABELS, MODEL, FEATURE_METHODS,
                                    ALL_METHODS_COMBINATIONS)

# __________QL-AGENT INITIALIZATION___________
AGENT = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS,
                        lr_min=0.001, lr_max=0.1, df=0.999, eps=1.0, eps_min=0.1,
                        eps_decay=0.001, decay=0.05, window_size=50,
                        total_episodes=MAX_EPISODES)

# __________GET INIIAL MODEL ACCURACY____________
# To set the initial accuracy.
ENV.get_initial_accuracy(DATA, LABELS, MODEL, num_folds=5)
print(f"Initial Accuracy: {ENV.initial_accuracy}")
print("-"*50)

#___________OUTPUTS & VISUALIZATIONS____________________

# Get outputs
OUTPUTS = train_evaluate_agent(ENV, AGENT, ALL_METHODS_COMBINATIONS,
                                MAX_EPISODES)
# Unpack outputs
episode_rewards, episode_accuracies, top_combinations, q_table = OUTPUTS

# Visualize the results of the Q-Learning agent.
VISUALIZE_OUTPUTS = visualize_results(episode_rewards, episode_accuracies,
                                      q_table, NUM_ACTIONS, top_combinations)

"""#**CIFAR10 DATASET**



*   **Problem Type:** Image Classification
*   **Model Used:** Convolution Neural Network (CNN)


"""

# ______________________CIFAR10 DATASET_________________________________________

from keras.datasets import cifar10
# Load CIFAR-10 dataset
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
# Get the total lengths
num_train = len(train_data)
num_test = len(test_data)
# Define subset sizes
num_train_samples = min(10000, num_train)
num_test_samples = min(2000, num_test)
# Take subsets from both data and labels
train_data_subset = train_data[:num_train_samples]
train_labels_subset = train_labels[:num_train_samples]
test_data_subset = test_data[:num_test_samples]
test_labels_subset = test_labels[:num_test_samples]
# Flatten the image data
train_data_flattened = train_data_subset.reshape((train_data_subset.shape[0], -1))
test_data_flattened = test_data_subset.reshape((test_data_subset.shape[0], -1))
train_labels_flattened = train_labels_subset.flatten()
test_labels_flattened = test_labels_subset.flatten()

DATA = train_data_flattened
LABELS = train_labels_flattened

print(f"Train Data Shape: {DATA.shape}")
print(f"Train Label Shape: {LABELS.shape}")
print(f"Test Data Shape: {DATA.shape}")
print(f"Test Label Shape: {LABELS.shape}")
print("-"*50)

# __________MODULE CONNECTION_________
selector = MODEL_SELECTOR()
# Select the problem type
problem_type = selector.select_problem_type()
model_options = selector.select_model(problem_type)
print(f"Select a suitable model for {problem_type}: {model_options}")
# User inputs the selected model
SELECTED_MODEL = input("Enter your selected model from the list above: ")

GET_MODEL = MODEL_BUILDER(SELECTED_MODEL)
MODEL = GET_MODEL.build_model()
FEATURE_METHODS_DIC = ALGORITHM_METHODS[SELECTED_MODEL]
FEATURE_METHODS = list(FEATURE_METHODS_DIC.keys())
ALL_METHODS_COMBINATIONS = COMBO_GENERATOR(FEATURE_METHODS).generate_combos()
if not ALL_METHODS_COMBINATIONS:
    raise ValueError("No feature method combinations were generated.")

print(f"========================== MODEL & FIEATURES SUMMARY ==========================")
print(f"\nSELECTED MODEL:")
print(f"{MODEL}\n")
print("FEATURE_METHODS:")
for idx, method in enumerate(FEATURE_METHODS, start=1):
    print(f"{idx}. {method}")
print("\nFEATURE_METHODS COMBINATIONS:")
for idx, combo in enumerate(ALL_METHODS_COMBINATIONS, start=1):
    print(f"    {idx}. {combo}")

print("-"*50)

# __________REQUIRED PARAMETERS________
NUM_ACTIONS = len(ALL_METHODS_COMBINATIONS)
NUM_STATES = NUM_ACTIONS + 1
MAX_EPISODES = 10

# __________ENVIRONMENT INITIALIZATION________
ENV = FeatureEngineeringEnvironment(DATA, LABELS, MODEL, FEATURE_METHODS,
                                    ALL_METHODS_COMBINATIONS)

# __________QL-AGENT INITIALIZATION___________
AGENT = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS,
                        lr_min=0.001, lr_max=0.1, df=0.999, eps=1.0, eps_min=0.1,
                        eps_decay=0.001, decay=0.05, window_size=50,
                        total_episodes=MAX_EPISODES)

# __________GET INIIAL MODEL ACCURACY____________
# To set the initial accuracy.
ENV.get_initial_accuracy(DATA, LABELS, MODEL, num_folds=5)
print(f"Initial Accuracy: {ENV.initial_accuracy}")
print("-"*50)

#___________OUTPUTS & VISUALIZATIONS____________________

# Get outputs
OUTPUTS = train_evaluate_agent(ENV, AGENT, ALL_METHODS_COMBINATIONS,
                                MAX_EPISODES)
# Unpack outputs
episode_rewards, episode_accuracies, top_combinations, q_table = OUTPUTS

# Visualize the results of the Q-Learning agent.
VISUALIZE_OUTPUTS = visualize_results(episode_rewards, episode_accuracies,
                                      q_table, NUM_ACTIONS, top_combinations)