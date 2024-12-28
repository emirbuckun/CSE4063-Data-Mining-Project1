from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC

def create_models():
    """Create all models"""
    models = {
        'Decision Tree (Gini)': DecisionTreeClassifier(
            criterion='gini',
            random_state=42,
            max_depth=10,
            class_weight='balanced'
        ),
        'Decision Tree (Entropy)': DecisionTreeClassifier(
            criterion='entropy', 
            random_state=42,
            max_depth=10,
            class_weight='balanced'
        ),
        'Naive Bayes (Bernoulli)': BernoulliNB(),
        'Naive Bayes (Complement)': ComplementNB(
            norm=True
        ),
        'SVM': LinearSVC(
            random_state=42,
            class_weight='balanced',
        ),
        'Neural Network (1 hidden layer)': MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ),
        'Neural Network (2 hidden layers)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        )
    }
    return models

def create_ensemble_models():
    """Create ensemble models"""
    simple_tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        class_weight='balanced'
    )
    
    ensemble_models = {
        'Bagging': BaggingClassifier(
            estimator=simple_tree,
            random_state=42
        ),
        'Boosting': AdaBoostClassifier(
            estimator=simple_tree,
            random_state=42,
            n_estimators=10,
            algorithm='SAMME'
        )
    }
    return ensemble_models