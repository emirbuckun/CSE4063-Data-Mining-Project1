import pandas as pd
from preprocessing import load_and_explore_data, preprocess_data
from models import create_models, create_ensemble_models
from evaluation import evaluate_model, plot_comparison
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def main():
    try:
        # Load the dataset
        print("Loading dataset...")
        data = pd.read_csv('data/dataset5.csv')
        
        # Separate features and target variable
        features = [
            'user_reputation', 'reply_count', 'thumbs_up', 
            'thumbs_down', 'best_score', 'text'
        ]
        target = 'stars'
        
        X = data[features]
        y = data[target]
        
        # Explore the data
        X, y = load_and_explore_data(X, y)
        
        # Preprocess the data
        print("\nPreprocessing data...")
        X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
        
        # Train models
        print("\nTraining models...")
        models = create_models()
        results = {}
        
        for name, model in models.items():
            try:
                print(f"\nTraining {name}...")
                model.fit(X_train, y_train)
                results[name] = evaluate_model(model, X_test, y_test, name)
                input(f"\nPress Enter to continue after viewing {name} results...")
            except Exception as e:
                print(f"Error: An issue occurred while training the {name} model:")
                print(str(e))
        
        # Cross-validation for one classifier
        print("\nPerforming cross-validation for Decision Tree (Gini)...")
        dt_gini = models['Decision Tree (Gini)']
        cv_scores = cross_val_score(dt_gini, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean cross-validation score: {cv_scores.mean()}")
        input(f"\nPress Enter to continue after viewing cross-validation results...")
        
        # Train ensemble models
        print("\nTraining ensemble models...")
        try:
            ensemble_models = create_ensemble_models()
            for name, model in ensemble_models.items():
                print(f"\nTraining {name}...")
                model.fit(X_train, y_train)
                results[name] = evaluate_model(model, X_test, y_test, name)
                input(f"\nPress Enter to continue after viewing {name} results...")
        except Exception as e:
            print("Error: An issue occurred while creating ensemble models:")
            print(str(e))
        
        # Compare results
        if results:
            print("\nComparing all models...")
            plot_comparison(results)
            
    except Exception as e:
        print("Error: An issue occurred while running the program:")
        print(str(e))

if __name__ == "__main__":
    main()
    print("\nProgram completed successfully. Press Enter to exit...")
    input()