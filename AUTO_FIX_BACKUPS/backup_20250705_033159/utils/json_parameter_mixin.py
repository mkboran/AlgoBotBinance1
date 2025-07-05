
class JSONParameterLoaderMixin:
    """💎 JSON tabanlı parametre yükleme mixin'i
    
    Bu mixin'i strategy sınıflarına ekleyerek JSON'dan
    otomatik parametre yükleme özelliği kazandırabilirsiniz.
    """
    
    def load_optimized_parameters(self, strategy_name: str = None) -> Dict[str, Any]:
        """📖 JSON dosyasından optimize edilmiş parametreleri yükle"""
        
        import json
        from pathlib import Path
        
        if strategy_name is None:
            strategy_name = getattr(self, 'strategy_name', 'unknown')
        
        # JSON dosya yolu
        json_path = Path("optimization/results") / f"{strategy_name}_best_params.json"
        
        if not json_path.exists():
            return {}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            parameters = data.get("parameters", {})
            
            # Logging (optional)
            if hasattr(self, 'logger'):
                self.logger.info(f"📖 {strategy_name} optimized parameters loaded: {len(parameters)} params")
            
            return parameters
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ Error loading optimized parameters: {e}")
            return {}
    
    def update_parameters_from_json(self, strategy_name: str = None) -> bool:
        """🔄 JSON'dan parametreleri yükleyip sınıf attribute'larını güncelle"""
        
        parameters = self.load_optimized_parameters(strategy_name)
        
        if not parameters:
            return False
        
        # Sınıf attribute'larını güncelle
        updated_count = 0
        for param_name, param_value in parameters.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
                updated_count += 1
        
        if hasattr(self, 'logger'):
            self.logger.info(f"🔄 Updated {updated_count} parameters from JSON")
        
        return updated_count > 0
