import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Dict, Union, Iterable
from datetime import datetime, timedelta


AssetType = Literal["equity", "index", "fund"]


class DataProvider(ABC):
    """
    Clase abstracta de proveedor que garantiza la homogeneidada en las siguientes clases.
    Todo proveedor debe tener 'price_history'.
    Esto admitirá cualquier proveedor de datos siempre que implemente este método.

    Para todos los AssetType y proveedores usaremos la columna "Adj Close" price salvo que se indique lo contrario.

    Tipos de activos soportados:
    1. equity: acciones
    2. index: índices bursátiles
    3. crypto: criptomonedas
    4. fund: fondos de inversión
    """

    name: str = "abstract" #el nombre del proveedor

    @abstractmethod
    def price_history(
        self,
        symbols: Union[str, Iterable[str]], #el código del ticker o una lista
        periods: int=252,
        asset_type: Literal["equity", "index", "crypto", "fund"] = "equity",
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Obtiene el historial de precios para los símbolos dados.
        Args:
            symbols: cadena o lista de cadenas de símbolos.
            periods: número de días hábiles a recuperar (por defecto 252).
            asset_type: tipo de activo: 'equity', 'index', 'crypto', 'fund'.
            **kwargs: argumentos específicos del proveedor.
        Returns:
            dict de DataFrames con los precios históricos para cada símbolo.
        """
        ...




class YahooProvider(DataProvider):
    ''' Proveedor de datos usando Yahoo Finance a través de yfinance.
    Extraerá un diccionario str:OHLCVA.
    '''
    name = "yfinance"

    def __init__(self):
        ''' Inicializa el proveedor de Yahoo Finance.'''
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except Exception:
            self.yf = None
            self.available = False
            print("⚠️ Advertencia: 'yfinance' no está instalado o no pudo importarse.")

    def price_history(
        self,
        symbols: Union[str, Iterable[str]],
        periods: int = 252,
        asset_type: AssetType = "equity",
        **kwargs
    ) -> pd.DataFrame:
        
        # --- 1) Calcular rango de fechas ---
        end_dt =datetime.now()
        start_dt = end_dt - timedelta(days=int(periods * 1.5)) #buffer para días no laborables
        
        # --- 2) Descargar datos de Yahoo ---
        raw = self.yf.download(symbols, start=start_dt, end=end_dt,auto_adjust=False, progress=False, **kwargs)   
        if periods:
             raw = raw.tail(periods)
        
        
        # --- 3) Normalizar y convertir a dict ---
        dict_final = {}
        if isinstance(symbols, str):
            symbols = [symbols]

        for s in symbols:
            cols = [
                ("Open", s),
                ("High", s),
                ("Low", s),
                ("Close", s),
                ("Adj Close", s),
                ("Volume", s),
            ]
            existing = [c for c in cols if c in raw.columns]
            df_s = raw.loc[:, existing].copy()
            df_s.columns = [c[0].lower().replace(" ", "_") for c in df_s.columns]

            dict_final[s] = df_s
        
        
        # --- 4) Añadir metadatos ---
        for sym, df in dict_final.items():
            df.attrs['provider'] = self.name
            df.attrs['symbol'] = sym
            df.attrs['asset_type'] = asset_type

        return dict_final


class AlphaVantageProvider(DataProvider):
    '''Proveedor de datos usando Alpha Vantage.
    Extraerá un diccionario str:OHLCVA.
    '''
    name = "alpha_vantage"

    def __init__(self, api_key: Optional[str] = None):
        '''Inicializa el proveedor de Alpha Vantage.'''
        try:
            from alpha_vantage.timeseries import TimeSeries
            self.TimeSeries = TimeSeries
            self.available = True
        except Exception:
            self.TimeSeries = None
            self.available = False
            print("⚠️ Advertencia: 'alpha_vantage' no está instalado o no pudo importarse.")
            return

        # --- Guarda la API key ---
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            print("⚠️ No se proporcionó clave de API de Alpha Vantage. Usa un argumento o variable de entorno ALPHAVANTAGE_API_KEY.")
        else:
            self.ts = self.TimeSeries(key=self.api_key, output_format='pandas')

    def price_history(
        self,
        symbols: Union[str, Iterable[str]],
        periods: int = 252,
        asset_type: AssetType = "equity",
        **kwargs
    ) -> dict[str, pd.DataFrame]:

        # --- 1) Normalizar la lista de símbolos ---
        if isinstance(symbols, str):
            symbols = [symbols]

        dict_final = {}

        # --- 2) Descargar cada símbolo individualmente ---
        for s in symbols:
            try:
                # En Alpha Vantage el endpoint de acciones usa get_daily_adjusted
                df, meta = self.ts.get_daily_adjusted(symbol=s, outputsize='full', **kwargs)

                # Ordenar por fecha ascendente
                df = df.sort_index()

                # Renombrar columnas para mantener consistencia
                df = df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. adjusted close': 'adj_close',
                    '6. volume': 'volume'
                })

                # Quedarse con los últimos N períodos
                if periods:
                    df = df.tail(periods)

                # Guardar sólo la columna de precios ajustados
                price_series = df.copy()
                dict_final[s] = price_series

                # --- 3) Añadir metadatos ---
                price_series.attrs['provider'] = self.name
                price_series.attrs['symbol'] = s
                price_series.attrs['asset_type'] = asset_type
                price_series.attrs['meta'] = meta

            except Exception as e:
                print(f"⚠️ Error descargando {s} desde Alpha Vantage: {e}")
                dict_final[s] = pd.Series(dtype=float)

        return dict_final

