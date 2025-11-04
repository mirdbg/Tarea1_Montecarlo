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
    data: PriceSeries = field(default_factory=PriceSeries)
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
            self.symbol = getattr(self.data, "symbol", "DESCONOCIDO") or "DESCONOCIDO"

        # 1) Obtener el DataFrame interno
        df = getattr(self.data, "data", None)
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("❌ 'data' debe ser una PriceSeries con un atributo .data (DataFrame).")

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
        self.data.data = df.sort_index()

    # ---------- Cálculo de retornos ----------
    def _log_returns(self) -> pd.Series:
        """
        Calcula los retornos logarítmicos diarios a partir de la columna 'price'
        de la PriceSeries interna (self.data.data). Devuelve una Serie sin NaN.
        """
        return np.log(self.data.data["price"]).diff().dropna()

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
        price0 = float(self.data.data["price"].iloc[-1]) if start_price is None else float(start_price)

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
    def plot_history(
        self,
        include_sims: bool = True,
        max_paths: int = 50,
        path: Optional[str] = None,
    ):
        """
        Grafica el histórico (self.data) y, si hay simulación en self._last_sim,
        sobrepone las trayectorias de Monte Carlo partiendo de la última fecha histórica.
        """
        df = self.data.data  # DataFrame con la columna 'price'
        if df.empty:
            raise ValueError("No hay datos históricos para graficar.")

        # Figura
        plt.figure()

        # 1) Histórico
        plt.plot(df.index, df["price"], label="Histórico", linewidth=2)

        # 2) Simulaciones (si existen y se pide)
        if include_sims and (self._last_sim is not None):
            sims = self._last_sim  # shape: (days+1, n_sims)
            if sims.ndim != 2:
                raise ValueError("Simulación inválida almacenada en self._last_sim.")

            # Fechas futuras: mismo punto inicial (día 0 = última fecha histórica),
            # y luego días hábiles hacia adelante para las filas 1..days
            last_dt = df.index[-1]
            days = sims.shape[0] - 1
            future_idx = pd.bdate_range(last_dt, periods=days + 1)  # incluye last_dt como 1º
            # Aseguramos longitud = rows de sims
            if len(future_idx) != sims.shape[0]:
                # fallback por si el calendario difiere
                future_idx = pd.date_range(last_dt, periods=sims.shape[0], freq="B")

            # Pintamos hasta max_paths caminos
            k = min(max_paths, sims.shape[1])
            for i in range(k):
                plt.plot(future_idx, sims[:, i], alpha=0.25)

            plt.legend(loc="best")

        plt.title(f"Evolución histórica y simulaciones - {self.symbol}")
        plt.xlabel("Fecha")
        plt.ylabel("Precio")

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


    def plot_simulations(
        self,
        prices: Optional[np.ndarray] = None,
        max_paths: int = 50,
        path: Optional[str] = None,
    ):
        """
        Grafica trayectorias simuladas. Si no se pasa `prices`,
        usa la última simulación cacheada en `self._last_sim`.
        """
        sims = prices if prices is not None else self._last_sim
        if sims is None:
            raise ValueError("No hay simulaciones disponibles. Ejecuta monte_carlo() primero.")
        if sims.ndim != 2:
            raise ValueError("Se esperaba un array 2D (days+1, n_sims).")

        # Fechado relativo (0..days) o por fechas hábiles desde la última fecha histórica
        last_dt = self.data.data.index[-1]
        days = sims.shape[0] - 1
        future_idx = pd.bdate_range(last_dt, periods=days + 1)

        plt.figure()
        k = min(max_paths, sims.shape[1])
        for i in range(k):
            plt.plot(future_idx, sims[:, i], alpha=0.25)
        plt.title(f"Simulaciones Monte Carlo - {self.symbol}")
        plt.xlabel("Fecha")
        plt.ylabel("Precio")

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


    def plot_final_hist(
        self,
        finals: Optional[np.ndarray] = None,
        bins: int = 50,
        path: Optional[str] = None,
    ):
        """
        Histograma de precios finales simulados. Si no se pasa `finals`,
        usa la última simulación cacheada.
        """
        if finals is None:
            if self._last_sim is None:
                raise ValueError("No hay simulaciones disponibles. Ejecuta monte_carlo() primero.")
            finals = self._last_sim[-1, :]

        plt.figure()
        plt.hist(finals, bins=bins)
        plt.title(f"Distribución de valor final - {self.symbol}")
        plt.xlabel("Precio final")
        plt.ylabel("Frecuencia")

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
