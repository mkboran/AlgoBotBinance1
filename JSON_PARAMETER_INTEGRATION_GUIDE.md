
# 💎 JSON PARAMETER SYSTEM INTEGRATION GUIDE

## 📖 Strateji Parametrelerini JSON'dan Yükleme

### 1. Yeni Strateji Oluştururken:

```python
class YourStrategy:
    def __init__(self, portfolio: Portfolio, **kwargs):
        self.portfolio = portfolio
        self.strategy_name = "your_strategy"
        
        # Varsayılan parametreler
        self.default_parameters = {
            "param1": 10,
            "param2": 0.5
        }
        
        # JSON'dan parametreleri yükle
        self._load_parameters_from_json()
        
        # Manuel override'ları uygula
        self._apply_manual_overrides(kwargs)
```

### 2. Optimizasyon Sonrası Parametre Kaydetme:

```python
# Optimizasyon script'inde:
best_parameters = {
    "param1": 15,
    "param2": 0.7
}

optimization_metrics = {
    "best_score": 0.85,
    "sharpe_ratio": 2.5,
    "max_drawdown": 0.08
}

# JSON'a kaydet
from json_parameter_system import JSONParameterManager
manager = JSONParameterManager()
manager.save_optimization_results(
    strategy_name="your_strategy",
    best_parameters=best_parameters,
    optimization_metrics=optimization_metrics
)
```

### 3. Mevcut Stratejiyi JSON Sistemi için Güncelleme:

```python
# Eski yöntem (KALDIRIN):
# if hasattr(settings, 'OPTIMIZED_PARAM1'):
#     self.param1 = settings.OPTIMIZED_PARAM1

# Yeni yöntem (EKLEYIN):
def _load_parameters_from_json(self):
    json_path = Path("optimization/results") / f"{self.strategy_name}_best_params.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        parameters = data.get("parameters", {})
        for param_name, param_value in parameters.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
```

## 🎯 Avantajları:

✅ %100 güvenilir (kaynak kodu değişmez)
✅ Versiyon kontrolü dostu
✅ Otomatik backup
✅ Metadata tracking
✅ Hata tolere edebilir
✅ Manuel override desteği

## 🚀 Komutlar:

```bash
# Optimizasyon sonuçlarını kaydet:
python json_parameter_system.py save --strategy momentum --params results.json

# Strateji parametrelerini güncelle:
python json_parameter_system.py update --strategy momentum

# Tüm stratejileri güncelle:
python json_parameter_system.py update-all

# Parametre dosyalarını listele:
python json_parameter_system.py list

# Parametre dosyasını incele:
python json_parameter_system.py inspect --strategy momentum
```
