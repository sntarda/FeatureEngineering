# IMPORT LIBRARIES:
from os import get_inheritable
import numpy as np
import pandas as pd
import itertools
import random
import logging
import Augmentor
import heapq
import math
import time
from scipy import sparse
import matplotlib.pyplot as plt
from multiprocessing import process
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from itertools import chain, combinations

# Suppress specific warnings (optional)
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning  # Ignore ConvergenceWarnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore FutureWarnings
warnings.filterwarnings('ignore', category=UserWarning)    # Ignore UserWarnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)  # Ignore ConvergenceWarnings

# IMPORT MODULE/S:
import QLAgentModule
from QLAgentModule import MODEL_SELECTOR, MODEL_BUILDER, METHODS_REQUIRE_LABELS, ALGORITHM_METHODS, COMBO_GENERATOR, DATA_ENGINEERING
#METHODS_REQUIRE_LABELS = QLAgentModule.METHODS_REQUIRE_LABELS


# _________________________AGENT ENVIRONMENT SETUP______________________________

class FeatureEngineeringEnvironment:
    def __init__(self, data, labels, model, feature_methods, random_combinations):

        #________IMPORTANT DATA CHECKS________________
        # Data Checks - ensure data and labels are reshaped properly
        data_to_proc, labels_to_proc = self.check_and_reshape(data, labels) 

        #________ENVIRONMENT VARIABLES INITIALIZATION_______
        self.data = data_to_proc  # Use the checked data
        self.labels = labels_to_proc  # Use the checked labels
        self.model = model
        self.feature_methods = feature_methods
        self.random_combinations = random_combinations
        self.initial_accuracy = None
        self.current_accuracy = None  # Fixed: it should be None as initial value
        self.initial_step = 0
        self.max_steps = 10

        #________LOGGER INTIALIZATION___________________
        self.logger = logging.getLogger('FeatureEngineeringEnvironment')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        #________DATA_COLLECTION INITALIZATION__________
        self.last_eval_results = None  # To store the results of the last evaluation
        self.action_history = []  # To store the history of actions taken
        self.reward_history = []  # To store the history of rewards received
        self.accuracy_history = []  # To store the history of accuracy metrics
        self.step_history = []  # To store the history of steps
        self.processed_data_history = []  # To store the history of processed data

    def get_initial_accuracy(self, data, labels, model, num_folds=5):
        """Evaluate the model on the provided original data and labels using cross-validation."""
        # Check if the number of samples in data and labels match
        processed_data, labelsToProcess = self.check_and_reshape(data, labels)
        # Perform cross-validation
        cv_scores = cross_val_score(model, data, labels, cv=num_folds)
        # Calculate the mean accuracy from cross-validation
        accuracy = cv_scores.mean()
        self.initial_accuracy = accuracy
        print(f"Initial accuracy (cross-validation mean): {accuracy:.4f}")
        return accuracy

    def apply_feature_methods(self, dataTrain, labels, feature_methods, random_combinations):
            """
            Apply the list of feature engineering methods to the data.
            :param data: The dataset to process.
            :param feature_methods: A list of feature engineering methods to apply.
            :param random_combinations: A list of selected feature engineering methods from random combinations.
            :return: The processed dataset.
            """
            print("shape of the input data is: ", dataTrain.shape)
            self.logger.info("Starting feature engineering process...")
            processed_data_trainset = [] # To avoid changing the original data
            # Import the correct configuration
            print(f"Random combinations ({len(random_combinations)}) :  {random_combinations}")
            #print(len(random_combinations))
            cc = 1
            for combination in random_combinations:
                print(f"========================== Random combination {cc}:{combination} ==========================")
                cc += 1
                processed_data = dataTrain.copy()
                for method in feature_methods:
                    #print("method: ", method)
                    print("Shape after feature engineering - Train:", processed_data.shape)
                    # Check if the method is in the list of random combinations
                    if method in combination:
                        self.logger.info(f"Applying feature engineering for method: {method}")
                        data_engineer = DATA_ENGINEERING()  # Instance of the DataEngineering class
                        methods_with_labels = METHODS_REQUIRE_LABELS  # Methods that require labels
                        if data_engineer:
                            # Apply the function. This assumes that your method functions are designed
                            # to process the data and return the transformed data.
                            print(f"Applying feature engineering for method: {method}")
                            if method in methods_with_labels:
                                try:
                                    data2train = data_engineer.engineer_data(processed_data, labels, method, transform_only=False)
                                    print("Shape of the data now is: ", data2train.shape)

                                    # Update the data to process
                                    processed_data = data2train.copy()
                                except Exception as e:
                                    self.logger.error(f"Method {method} failed: {e}")
                            else:
                                try:
                                    data2train = data_engineer.engineer_data(processed_data, method, transform_only=False)
                                    print("Shape of the data now is: ", data2train.shape)

                                    # Update the data to process
                                    processed_data = data2train.copy()
                                except Exception as e:
                                    self.logger.error(f"Method {method} failed: {e}")

                        else:
                            self.logger.error(f"Method {method} not recognized or not implemented.")
                processed_data_trainset.append(processed_data)
                self.processed_data_history.append((combinations, processed_data))

            print("=================================================================================================================================")
            for i, proc_data in enumerate(processed_data_trainset):
                print(f"Data processed after applying combination number {i+1}, shape is {proc_data.shape}")

            self.logger.info("Feature engineering completed.")

            return processed_data_trainset


    def train_evaluate_model(self, model, processed_data, labels):
        """Evaluate the model using cross-validation."""

        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
            'f1_score': 'f1_macro'
        }
        # Initialize an empty list to store evaluation results
        evaluation_results = []
        # Check and reshape the processed data and labels
        processed_data, labels = self.check_and_reshape(processed_data, labels)
        # Perform cross-validation and collect the scores
        scores = cross_validate(model, processed_data, labels, cv=5, scoring=scoring_metrics)
        # Calculate the mean of each metric across the cross-validation folds
        mean_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring_metrics}
        # Append the mean scores to the evaluation results
        evaluation_results.append(mean_scores)
        # Sort the results based on accuracy in descending order
        evaluation_results = sorted(evaluation_results, key=lambda x: x['accuracy'], reverse=True)
        # Update the last evaluation results
        self.last_eval_results = evaluation_results
        # Optionally log the evaluation results
        self.logger.info(f"Evaluation results: {evaluation_results}")
        # Return the evaluation results
        return evaluation_results

    def step(self, action_combination):

        #___IMPORTANT STATE CHECK____
        if self.current_step >= self.max_steps:
            return None, 0, True, None
        # Check if the action combination is valid
        if not all(method in self.feature_methods for method in action_combination):
            self.logger.error("Invalid action combination")
            return None, 0, True, None
        # Check if the action combination has already been taken
        if action_combination in self.action_history:
            self.logger.error("Action combination already taken")
            return None, 0, True, None
        #___APPLY FEATURE METHODS___
        # Apply the feature methods to get transfomred data
        self.apply_feature_methods(self.data, self.labels, self.feature_methods, self.random_combinations)
        current_processed_data = self.processed_data_history[-1][1]
        #___TRAIN & EVALUATE MODEL___     
        # Get the selected model
        evaluation_results = self.train_evaluate_model(self.model, current_processed_data, self.labels)
        # Calculate the reward based on the evaluation results
        evaluation_metrics = evaluation_results[-1]
        weights = {'accuracy': 0.25, 'precision': 0.25, 'recall': 0.25, 'f1_score': 0.25}
        reward = sum(weights[metric] * evaluation_metrics[metric] for metric in weights)
        accuracy = evaluation_metrics['accuracy']

        # Update the current accuracy
        self.current_accuracy = accuracy
        # Update the current step
        self.current_step += 1
        # Check if the current step is the last step
        done = self.current_step == self.max_steps
        # Logging the action and reward
        self.logger.info(f"Action taken: {action_combination}, Reward received: {reward}")

        # Store the action and the reward in history for analysis
        self.action_history.append(action_combination)
        self.reward_history.append(reward)
        self.accuracy_history.append(self.current_accuracy)

        # Existing step method...
        return reward, done, evaluation_results

    def reset(self):
        # Resetting the history when environment is reset
        self.action_history.clear()
        self.reward_history.clear()
        self.accuracy_history.clear()
        self.processed_data_history.clear()
        # Existing reset method code...
        self.current_step = self.initial_step
        self.current_accuracy = self.initial_accuracy

    # Additional methods to access the history for analysis (OPTIONAL based on use)
    def get_action_history(self):
        return self.action_history
    def get_reward_history(self):
        return self.reward_history
    def get_accuracy_history(self):
        return self.accuracy_history
    def get_last_eval_results(self):
        return self.last_eval_results

    def check_and_reshape(self, data, labels):
        # Check if data is sparse and convert to dense if necessary
        if sparse.issparse(data):
            data = data.toarray()

        # Now, the method will work for both DataFrames and NumPy arrays
        if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
            data = np.array(data)  # Ensures data is always in NumPy array format
        else:
            raise TypeError(f"Data type not supported: {type(data)}")

        data_samples = data.shape[0]
        label_samples = labels.shape[0]

        # Check if the number of samples matches
        if data_samples != label_samples:
            # Check if labels can be reshaped to match data
            if labels.size == data_samples:
                new_shape = (data_samples, 1)  # Explicitly reshape to 2D array with one column
                labels = labels.reshape(new_shape)
            else:
                # Raise an error showing the mismatch in shapes
                raise ValueError(f"Data and labels shape mismatch. Data shape: {data.shape}, Labels shape: {labels.shape}")

        return data, labels

# _________________________Q-LEARNING AGENT DESIGN______________________________

class QLearningAgent:
    def __init__(self, num_states, num_actions, lr_min, lr_max, df,
                 eps, eps_min, eps_decay, decay, window_size, total_episodes):
      
        #________AGENT VARIABLES INITIALIZATION_______
        self.num_actions = num_actions
        self.num_states = num_states
        self.q_table = np.full((self.num_states, self.num_actions), 100)
        self.learning_rate_min = lr_min
        self.learning_rate_max = lr_max
        self.learning_rate = lr_min  # Initial learning rate
        self.df = df  # Discount factor
        self.epsilon_max = eps
        self.epsilon_min = eps_min
        self.epsilon_decay_rate = eps_decay
        self.epsilon = self.epsilon_max  # Initial epsilon value
        self.epsilon_decay = decay
        self.rewards = []
        self.window = window_size
        self.step = 0
        self.lr_cycle_length = 1000  # Adjust as needed
        self.current_state = None  # Variable to track the current state
        self.current_action = None  # Variable to track the current action
        self.current_episode = 0
        self.total_episodes = total_episodes  # Total number of training episodes

        #________LOGGER INTIALIZATION___________________
        self.logger = logging.getLogger('QLearningAgent')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def cyclic_learning_rate(self):
        """ 
        Adjust the learning rate following a sinusoidal pattern.
        This helps the agent to escape local minima in the learning process 
        and find the global minimum.
        """ 

        self.learning_rate = self.learning_rate_min + \
            (self.learning_rate_max - self.learning_rate_min) * \
            (np.sin(self.step / self.lr_cycle_length * np.pi) + 1) / 2
        self.step += 1
        return self.learning_rate

    def epsilon_greedy_decay(self, current_episode, total_episodes):
        """
        Adjust the decay of epsilon value used for exploration based on the episode number.
        As it learns more about the environment, it relies more on its learned knowledge (exploiting the best-known actions).
        The epsilon decay is to ensure that the amount of exploration decreases over time.
        """

        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                       math.exp(-self.epsilon_decay_rate * current_episode / total_episodes)
        # Ensure epsilon is within the valid range
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def select_action(self, current_state):
        # Select an action based on the current state
        self.epsilon_greedy_decay(self.current_episode, self.total_episodes)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.q_table[current_state])
        self.current_state = current_state
        self.current_action = action
        return action

    def update(self, current_state, action, reward, next_state):
        if not (0 <= current_state < self.num_states) or \
           not (0 <= action < self.num_actions) or \
           not (0 <= next_state < self.num_states):
            raise ValueError("Invalid state or action")

        future_reward = np.max(self.q_table[next_state])
        old_q_value = self.q_table[current_state, action]

        # Debugging prints
        print(f"future_reward: {future_reward}, old_q_value: {old_q_value}, reward: {reward}")

        updated_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.df * future_reward)

        # Check if updated_q_value is NaN
        if np.isnan(updated_q_value):
            print(f"Warning: updated_q_value is NaN! current_state: {current_state}, action: {action}, reward: {reward}, future_reward: {future_reward}")
            # Handle NaN case, e.g., by setting a default value or aborting the update
            updated_q_value = 0  # or some other default value

        self.q_table[current_state, action] = updated_q_value

# _________________________TRAIN & EVALYUATE AGENT_______________________________

def train_evaluate_agent(env, agent, random_combinations, max_episodes):
    episode_rewards = []
    episode_accuracies = []
    top_combinations = []

    for episode in range(max_episodes):
        agent.epsilon_greedy_decay(episode, max_episodes)
        env.reset()
        episode_reward = 0

        for state, action_comb in enumerate(random_combinations):
            if state >= agent.num_states or state >= env.max_steps:
                break

            action_idx = agent.select_action(state)
            reward, done, evaluation_results = env.step(action_comb)
            next_state = state + 1 if state + 1 < len(random_combinations) else state
            agent.update(state, action_idx, reward, next_state)

            episode_reward += reward
            current_accuracy = env.current_accuracy

            if len(top_combinations) < 3 or current_accuracy > min(top_combinations, key=lambda x: x[0])[0]:
                if len(top_combinations) >= 3:
                    top_combinations.remove(min(top_combinations, key=lambda x: x[0]))
                top_combinations.append((current_accuracy, action_comb, evaluation_results, episode + 1, state))

            agent.logger.info(f"Episode: {episode + 1}, State: {state}, Action: {action_comb}, Reward: {reward}, Accuracy: {current_accuracy}")

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_accuracies.append(env.current_accuracy)
        agent.logger.info(f"Episode {episode + 1} completed - Reward: {episode_reward}, Accuracy: {env.current_accuracy}")

    top_combinations.sort(reverse=True, key=lambda x: x[0])

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print("\n===== Training Summary =====\n")
    print(f"Average Reward: {avg_reward}")

    for idx, (accuracy, combination, eval_results, episode_num, state) in enumerate(top_combinations, start=1):
        formatted_accuracy = f"{accuracy*100:.2f}%"
        print(f"Top Combination #{idx} (Found in Episode {episode_num}, State {state}):")
        print(f"  Accuracy: {formatted_accuracy}")
        print(f"  Combination: {combination}")
        for result in eval_results:
            print(f"    - Eval Result: {result}")
        print("\n")

    return episode_rewards, episode_accuracies, top_combinations, agent.q_table


# _________________________OUTPUT & VISIUALIZATION______________________________

def visualize_results(episode_rewards, episode_accuracies, q_table, num_actions, top_combinations):
    """
    Visualizes the results of the Q-Learning agent's training process.
    :param episode_rewards: List of rewards obtained per episode.
    :param episode_accuracies: List of accuracies obtained per episode.
    :param q_table: The Q-table learned by the agent.
    :param num_actions: Number of actions in the Q-learning environment.
    """
    print("-"*50)
    print("Top Combinations:")
    for idx, (accuracy, combination, eval_results, episode_num, state) in enumerate(top_combinations, start=1):
        print(f"Top Combination #{idx}:"
              f"\n  Accuracy: {accuracy*100:.2f}%"
              f"\n  Combination: {combination}"
              f"\n  Eval Results: {eval_results}\n\n")

    # Print formatted Q-table
    print("Learned Q-Table:")
    print(pd.DataFrame(q_table, columns=list(range(num_actions))))

     
    print("-"*50)

    # Plot metrics over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot rewards
    ax1.set(title="Episode Rewards", ylabel="Reward")
    ax1.plot(episode_rewards, color='blue', label="Raw")
    ax1.plot(pd.Series(episode_rewards).rolling(5).mean(), color="green", label="Smoothed")
    ax1.legend()

    max_reward = np.max(episode_rewards)
    max_reward_ep = np.argmax(episode_rewards)
    ax1.text(max_reward_ep, max_reward, "Max Reward", ha="center")
    ax1.fill_between(range(len(episode_rewards)), episode_rewards, color="blue", alpha=0.3)

    # Plot accuracies
    ax2.set(title="Episode Accuracies", ylabel="Accuracy")
    ax2.plot(episode_accuracies, color="red", label="Raw")
    ax2.plot(pd.Series(episode_accuracies).rolling(5).mean(), color="orange", label="Smoothed")
    ax2.legend()

    fig.suptitle("Training Performance Over Time")
    fig.tight_layout()

    # Print formatted Q-table
    print("Learned Q-Table:")
    print(pd.DataFrame(q_table, columns=list(range(num_actions))))
    
    # Plot Q-table heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(q_table)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Q-Value", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(num_actions))
    ax.set_yticks(np.arange(len(q_table)))
    ax.set_xticklabels(range(num_actions))
    ax.set_yticklabels(range(len(q_table)))

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Q-Table Heatmap")
    fig.tight_layout()

    plt.show()
