from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n=== {model_name} Results ===")
    for metric, value in results.items():
        if metric != 'Confusion Matrix' and metric != 'Model':
            print(f"{metric}: {value:.4f}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    cm = results['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return results

def plot_comparison(results_dict):
    """Visualize model comparison results"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        plt.bar(models, values)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()
        plt.close()