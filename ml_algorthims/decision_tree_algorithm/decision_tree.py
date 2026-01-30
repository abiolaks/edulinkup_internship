"""
# # Understanding Decision Tree Classifier
# 
# This exercise demonstrate how to implement a decision tree classifier using scikit learn ml libray
# 
# ## Objectives
# * Uses the Iris dataset
# * understand the max_depth parameter
# * understand the splitting criteria parameter
# * visualize the decision tree.
# 
# A decision tree makes decisions by asking the best yes/no questions step by step untill it can confidently give an answer.
# 
# 
# - Max_depth : This parameter control  the number of levels of the tree depth. The tree depth is the maximum number of decisions levels from the root node(top) to the leaf node(bottom) in a decision tree.
# 
# - Splitting Criteria : This is the metric used to determine the best feature and the threshold to split the data at each node. we two spliting criteria which are Gini or Entropy. The criteria is set using the criterion parameter.
# 
# Gini measure how mixed the classes are a node, Gini =0, node is pure (only one class), Higher gini means more mixed classes.
# 
# * Entropy: is the information gain. It measures uncertainty before and after the split. High entropy high uncertainty, low entropy low uncertainty.
"""

# import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



# Load the dataset
iris = load_iris()
X = iris.data # Features
y = iris.target # Target variable


# Train test split
                                                    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the decision tree classifier using the following parameters:
"""
    criterion='gini', # Use Gini impurity as the splitting criterion
    max_depth=3,     # Limit the maximum depth of the tree to 3
    random_state=42 # Set a random seed for reproducibility
    """
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Fit the model to the training data
clf.fit(X_train, y_train)


# Make predictions on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")


# visualize the decision tree
"""
    plot_tree shows:
        - Feature used for splitting at each node
        - Threshold values for splits
        - Gini impurity at each node
        - Number of samples at each node
        - Predicted class for each leaf node

"""
plt.figure(figsize=(10,6))
plot_tree(clf, filled=True, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization (Max Depth = 3)")
plt.show()


