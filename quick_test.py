#!/usr/bin/env python3
"""
🚀 QUICK BACKTEST TEST
"""

import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path

async def quick_test():
    """Hızlı backtest testi"""
    
    print("🚀 QUICK BACKTEST TEST STARTING...")
    
    try:
        # 1. Veri dosyasını kontrol et
        data_file = "historical_data/BTCUSDT_15m_20240101_20241231.csv"
        data_path = Path(data_file)
        
        if not data_path.exists():
            print(f"❌ Data file not found: {data_file}")
            return
        
        print(f"✅ Data file found: {data_file}")
        
        # 2. Veriyi yükle ve kontrol et
        df = pd.read_csv(data_path)
        print(f"📊 Data loaded: {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        
        # 3. Tarih aralığını kontrol et
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # FORCE TIMEZONE NAIVE
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            df.set_index('timestamp', inplace=True)
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 3, 31)
            
            filtered_data = df.loc[
                (df.index >= start_date) & 
                (df.index <= end_date)
            ]
            
            print(f"📅 Filtered data: {len(filtered_data)} candles")
            print(f"📈 Period: {filtered_data.index[0]} to {filtered_data.index[-1]}")
            
            # 4. Basit analiz
            initial_price = filtered_data['close'].iloc[0]
            final_price = filtered_data['close'].iloc[-1]
            return_pct = (final_price - initial_price) / initial_price * 100
            
            print(f"💰 Initial price: ${initial_price:.2f}")
            print(f"💰 Final price: ${final_price:.2f}")
            print(f"📈 Buy & Hold return: {return_pct:+.2f}%")
            
            print("✅ QUICK TEST COMPLETED - Data looks good!")
            
        else:
            print("❌ No timestamp column found")
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
