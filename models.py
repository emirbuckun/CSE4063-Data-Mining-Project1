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
        'Naive Bayes (Bernoulli)': BernoulliNB(
            alpha=1.0,
            binarize=0.0
        ),
        'Naive Bayes (Complement)': ComplementNB(
            alpha=1.0,
            norm=True,
            force_alpha=True
        ),
        'SVM': LinearSVC(
            random_state=42,
            max_iter=2000,
            dual=False,
            C=1.0, # Regularization parameter
            class_weight='balanced',
            multi_class='ovr' # One-vs-Rest (OvR) strategy
        ),
        'Neural Network (1 hidden layer)': MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
        'Neural Network (2 hidden layers)': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    return models

def create_ensemble_models():
    """Create ensemble models"""
    simple_tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    
    ensemble_models = {
        'Bagging': BaggingClassifier(
            estimator=simple_tree,
            n_estimators=10,
            max_samples=0.7,
            max_features=0.7,
            random_state=42
        ),
        'Boosting': AdaBoostClassifier(
            estimator=simple_tree,
            n_estimators=10,
            learning_rate=0.1,
            random_state=42,
            algorithm='SAMME'
        )
    }
    return ensemble_models