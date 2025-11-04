from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Se asume que ya tienes definida tu clase PriceSeries en models.py
# con al menos: .data (pd.DataFrame con columna 'price') y .symbol (str)
from models import PriceSeries


@dataclass
class MonteCarloSimulation:
    """
    Simulador Monte Carlo (GBM) basado en una serie histórica de precios.

    Requiere:
      - data: PriceSeries con .data (DataFrame con índice DatetimeIndex y columna 'price')
              y .symbol (identificador del activo).
      - symbol: si no se proporciona, se toma de data.symbol.
    """
    price_series: PriceSeries = field(default_factory=PriceSeries)
    symbol: Optional[str] = None
    _last_sim: np.ndarray | None = None  # Almacena la última simulación realizada

    # ---------- Validación inicial ----------
    def __post_init__(self):
        """
        Validación automática tras crear el objeto y normalización de metadatos.
        - Toma el símbolo de la PriceSeries si no se pasa explícitamente.
        - Valida que el DataFrame interno tenga 'price' y DatetimeIndex.
        - Ordena por fecha.
        """
        # 0) Resolver símbolo por defecto desde la PriceSeries
        if self.symbol is None:
            # si la PriceSeries tiene symbol, úsalo; si no, pon "DESCONOCIDO"
            self.symbol = getattr(self.price_series, "symbol", "DESCONOCIDO") or "DESCONOCIDO"

        # 1) Obtener el DataFrame interno
        df = getattr(self.price_series, "data", None)
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("❌ 'price_series' debe ser una PriceSeries con un atributo .data (DataFrame).")

        # 2) Validaciones básicas del DataFrame
        if df.empty:
            raise ValueError("❌ La serie interna está vacía. Proporcione precios en PriceSeries.data.")
        if "price" not in df.columns:
            raise ValueError("❌ Se esperaba una columna 'price' en PriceSeries.data.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("❌ El índice de PriceSeries.data debe ser un pandas.DatetimeIndex.")
        if df["price"].isna().all():
            raise ValueError("❌ Todos los valores de 'price' son NaN.")

        # 3) Orden temporal asegurada (y re-asignación normalizada)
        self.price_series.data = df.sort_index()

    # ---------- Cálculo de retornos ----------
    def _log_returns(self) -> pd.Series:
        """
        Calcula los retornos logarítmicos diarios a partir de la columna 'price'
        de la PriceSeries interna (self.price_series.data). Devuelve una Serie sin NaN.
        """
        return np.log(self.price_series.data["price"]).diff().dropna()

    # ---------- Simulación Monte Carlo ----------
    def monte_carlo(
        self,
        days: int = 252,
        n_sims: int = 2000,
        seed: Optional[int] = 123,
        start_price: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simula trayectorias futuras de precios utilizando un proceso tipo GBM.
        Usa mu y sigma empíricos calculados a partir de los retornos logarítmicos históricos.

        Devuelve:
            np.ndarray con forma (days+1, n_sims) con los precios simulados.
        """
        # 1) Precio inicial = último precio real o el proporcionado
        price0 = float(self.price_series.data["price"].iloc[-1]) if start_price is None else float(start_price)

        # 2) Retornos logarítmicos históricos
        r = self._log_returns()
        if r.empty:
            raise ValueError("❌ Historial insuficiente para calcular retornos.")

        # 3) Estimación empírica de mu y sigma (diarios)
        mu = r.mean()
        sigma = r.std(ddof=1)
        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
            raise ValueError("❌ Estimaciones de deriva/volatilidad no válidas.")

        # 4) Generación de aleatorios N(0,1)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((days, n_sims))

        # 5) Incrementos de GBM en log-precios
        increments = (mu - 0.5 * sigma**2) + sigma * z

        # 6) Acumulación temporal (log-precios relativos)
        paths = np.vstack([
            np.zeros((1, n_sims)),         # Día 0: desplazamiento nulo
            np.cumsum(increments, axis=0)  # Suma acumulada de incrementos
        ])

        # 7) Conversión de log-precio a precio
        prices = price0 * np.exp(paths)

        # 8) Almacenar la última simulación
        self._last_sim = prices

        return prices

    # ---------- Distribución de precios finales ----------
    def final_values(self, prices: np.ndarray) -> np.ndarray:
        """
        Devuelve un vector con los precios finales (última fila)
        de cada una de las simulaciones.
        """
        if prices.ndim != 2:
            raise ValueError("❌ Se esperaba un array 2D con forma (days+1, n_sims).")
        return prices[-1, :]

    # ---------- Método combinado de resumen ----------
    def simulate_and_summarize(
        self,
        days: int = 252,
        n_sims: int = 2000,
        seed: Optional[int] = 123,
        start_price: Optional[float] = None,
        percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
    ) -> dict:
        """
        Ejecuta la simulación Monte Carlo y devuelve un resumen con:
          - 'prices': matriz (days+1, n_sims) con todas las trayectorias simuladas
          - 'finals': vector (n_sims,) con los precios finales
          - 'summary': diccionario con percentiles de precios finales
        """
        prices = self.monte_carlo(days=days, n_sims=n_sims, seed=seed, start_price=start_price)
        finals = self.final_values(prices)
        summary = {f"p{p}": float(np.percentile(finals, p)) for p in percentiles}
        return {"prices": prices, "finals": finals, "summary": summary}

        # ---------- Gráficas ----------
    def plot_history(self, path: Optional[str] = None):
        if self.price_series.data.empty:
            return
        plt.figure()
        self.price_series.data["price"].plot(title=f"Historical Price - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()

    def plot_simulations(self, prices: np.ndarray, max_paths: int = 50, path: Optional[str] = None):
        k = min(max_paths, prices.shape[1])
        plt.figure()
        for i in range(k):
            plt.plot(prices[:, i])
        plt.title(f"Monte Carlo Simulations - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()

    def plot_final_hist(self, prices: np.ndarray, bins: int = 50, path: Optional[str] = None):
        finals= self.final_values(prices)
        plt.figure()
        plt.hist(finals, bins=bins)
        plt.title(f"Terminal Value Distribution - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()