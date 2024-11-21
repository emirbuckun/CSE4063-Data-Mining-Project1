import pandas as pd
from preprocessing import load_and_explore_data, preprocess_data
from models import create_models, create_ensemble_models
from evaluation import evaluate_model, plot_comparison

def main():
    try:
        # Veri setini yükle
        print("Veri seti yükleniyor...")
        data = pd.read_csv('dataset5.csv')
        
        # Özellikleri ve hedef değişkeni ayır
        features = [
            'user_reputation', 'reply_count', 'thumbs_up', 
            'thumbs_down', 'best_score', 'text'
        ]
        target = 'stars'
        
        X = data[features]
        y = data[target]
        
        # Veriyi keşfet
        X, y = load_and_explore_data(X, y)
        
        # Veriyi ön işle
        print("\nVeri ön işleniyor...")
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(X, y)
        
        # Modelleri eğit
        print("\nModeller eğitiliyor...")
        models = create_models()
        results = {}
        
        for name, model in models.items():
            try:
                print(f"\n{name} eğitiliyor...")
                model.fit(X_train, y_train)
                results[name] = evaluate_model(model, X_test, y_test, name)
                input(f"\n{name} sonuçlarını gördükten sonra devam etmek için Enter'a basın...")
            except Exception as e:
                print(f"Hata: {name} modeli eğitilirken bir sorun oluştu:")
                print(str(e))
        
        # Ensemble modelleri eğit
        print("\nEnsemble modeller eğitiliyor...")
        try:
            ensemble_models = create_ensemble_models(None)
            for name, model in ensemble_models.items():
                print(f"\n{name} eğitiliyor...")
                model.fit(X_train, y_train)
                results[name] = evaluate_model(model, X_test, y_test, name)
                input(f"\n{name} sonuçlarını gördükten sonra devam etmek için Enter'a basın...")
        except Exception as e:
            print("Hata: Ensemble modeller oluşturulurken bir sorun oluştu:")
            print(str(e))
        
        # Sonuçları karşılaştır
        if results:
            print("\nTüm modeller karşılaştırılıyor...")
            plot_comparison(results)
            
    except Exception as e:
        print("Hata: Program çalışırken bir sorun oluştu:")
        print(str(e))

if __name__ == "__main__":
    main()