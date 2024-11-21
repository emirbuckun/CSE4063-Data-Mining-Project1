from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    """Model performansını değerlendir"""
    y_pred = model.predict(X_test)
    
    results = {
        'Model': model_name,
        'Doğruluk': accuracy_score(y_test, y_pred),
        'Kesinlik': precision_score(y_test, y_pred, average='weighted'),
        'Duyarlılık': recall_score(y_test, y_pred, average='weighted'),
        'F1-Skor': f1_score(y_test, y_pred, average='weighted'),
        'Karmaşıklık Matrisi': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n=== {model_name} Sonuçları ===")
    for metric, value in results.items():
        if metric != 'Karmaşıklık Matrisi' and metric != 'Model':
            print(f"{metric}: {value:.4f}")
    
    # Karmaşıklık matrisini görselleştir
    plt.figure(figsize=(10, 8))
    cm = results['Karmaşıklık Matrisi']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.show()
    plt.close()  # Figure'ı kapat
    
    return results

def plot_comparison(results_dict):
    """Model sonuçlarını karşılaştırmalı görselleştir"""
    metrics = ['Doğruluk', 'Kesinlik', 'Duyarlılık', 'F1-Skor']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        models = list(results_dict.keys())
        values = [results_dict[model][metric] for model in models]
        
        plt.bar(models, values)
        plt.title(f'Model Karşılaştırması - {metric}')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()
        plt.close()  # Figure'ı kapat