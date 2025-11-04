# --- pretty plotting with seaborn ---
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ========== Estilo ========== 
sns.set_theme(
    context="notebook",   # tamaños de fuente agradables
    style="whitegrid",    # fondo claro con grid sutil
)

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _format_dates(ax):
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))

def _currency_fmt(symbol: str = "€"):
    return FuncFormatter(lambda y, _: f"{y:,.2f}{symbol}".replace(",", "X").replace(".", ",").replace("X", "."))

# ========== MIXIN ==========
class PlotMixinSeaborn:
    symbol: str
    # price_series puede ser:
    # - DataFrame con columna 'price'
    # - Series de precios
    # - objeto con .to_dataframe() -> DataFrame con 'price'
    price_series: pd.DataFrame | pd.Series | object

    def _df(self) -> pd.DataFrame:
        obj = getattr(self, "price_series", None)
        if obj is None:
            return pd.DataFrame(columns=["price"])
        if isinstance(obj, pd.DataFrame):
            if "price" in obj.columns:
                return obj[["price"]]
            if "close" in obj.columns:
                return obj.rename(columns={"close": "price"})[["price"]]
            if obj.shape[1] == 1:
                return obj.rename(columns={obj.columns[0]: "price"})
            raise ValueError("El DataFrame no tiene 'price'/'close' ni única columna.")
        if isinstance(obj, pd.Series):
            return obj.to_frame(name="price")
        if hasattr(obj, "to_dataframe"):
            df = obj.to_dataframe()
            if "price" not in df.columns and "close" in df.columns:
                df = df.rename(columns={"close": "price"})
            if "price" not in df.columns:
                raise ValueError("to_dataframe() debe incluir 'price'.")
            return df
        raise TypeError("price_series no es DataFrame/Series ni wrapper compatible.")

    # -------- 1) Histórico bonito --------
    def plot_history(
        self,
        path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 4),
        ylabel: str = "Precio",
        currency_symbol: Optional[str] = None,
        show_rolling: Optional[int] = 20,
        title: Optional[str] = None,
        dpi: int = 150,
    ):
        df = self._df()
        if df.empty:
            return None, None

        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(x=df.index, y=df["price"], ax=ax, linewidth=1.8, label="Precio")

        if show_rolling and show_rolling > 1 and len(df) >= show_rolling:
            roll = df["price"].rolling(show_rolling).mean()
            sns.lineplot(x=roll.index, y=roll.values, ax=ax, linewidth=1.8, linestyle="--",
                         label=f"Media {show_rolling}")

        ax.set_title(title or f"Historical Price — {self.symbol}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(ylabel)
        if currency_symbol:
            ax.yaxis.set_major_formatter(_currency_fmt(currency_symbol))
        _format_dates(ax)
        ax.legend(frameon=False)
        fig.tight_layout()

        if path:
            _ensure_dir(path)
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        return fig, ax

    # -------- 2) Simulaciones bonitas --------
    def plot_simulations(
        self,
        prices: np.ndarray,
        max_paths: int = 50,
        index: Optional[Iterable] = None,     # si pasas fechas, se usan
        figsize: tuple[int, int] = (10, 5),
        percentiles: tuple[int, int, int] = (5, 50, 95),
        ylabel: str = "Precio simulado",
        title: Optional[str] = None,
        dpi: int = 150,
        path: Optional[str] = None,
    ):
        if prices is None or prices.size == 0:
            return None, None

        steps, n_sims = prices.shape
        k = min(max_paths, n_sims)
        x = np.arange(steps) if index is None else np.array(list(index))

        fig, ax = plt.subplots(figsize=figsize)

        # trayectorias individuales: usamos palette automática y baja opacidad
        for i in range(k):
            sns.lineplot(x=x, y=prices[:, i], ax=ax, linewidth=1.0, alpha=0.25)

        # percentiles
        p_low, p_med, p_high = np.percentile(prices, percentiles, axis=1)
        sns.lineplot(x=x, y=p_med, ax=ax, linewidth=2.2, label=f"P{percentiles[1]}")
        ax.fill_between(x, p_low, p_high, alpha=0.18, label=f"P{percentiles[0]}–P{percentiles[2]}")

        ax.set_title(title or f"Monte Carlo Simulations — {self.symbol}")
        ax.set_xlabel("Fecha" if index is not None else "Paso")
        ax.set_ylabel(ylabel)
        if index is not None:
            _format_dates(ax)
        ax.legend(frameon=False)
        fig.tight_layout()

        if path:
            _ensure_dir(path)
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        return fig, ax

    # -------- 3) HIST + SIMS (combinado en una sola figura) --------
    def plot_history_with_simulations(
        self,
        prices: np.ndarray,
        use_business_days: bool = True,
        max_paths: int = 50,
        percentiles: tuple[int, int, int] = (5, 50, 95),
        figsize: tuple[int, int] = (11, 5.5),
        currency_symbol: Optional[str] = None,
        title: Optional[str] = None,
        ylabel: str = "Precio",
        dpi: int = 150,
        path: Optional[str] = None,
    ):
        """
        Pintar histórico + simulaciones arrancando en el último punto del histórico.
        - prices: shape = (steps, n_sims), pasos de simulación hacia adelante.
        """
        df = self._df()
        if df.empty or prices is None or prices.size == 0:
            return None, None

        # Índice futuro para simulaciones
        last_dt = df.index[-1]
        steps, n_sims = prices.shape
        if use_business_days:
            future_index = pd.bdate_range(last_dt, periods=steps+1, inclusive="right")  # desde el día siguiente hábil
        else:
            future_index = pd.date_range(last_dt, periods=steps+1, inclusive="right", freq="D")

        # Armar figura
        fig, ax = plt.subplots(figsize=figsize)

        # Histórico
        sns.lineplot(x=df.index, y=df["price"], ax=ax, linewidth=1.9, label="Histórico")

        # Simulaciones: cada columna i es una trayectoria (partimos en future_index)
        k = min(max_paths, n_sims)
        for i in range(k):
            sns.lineplot(x=future_index, y=prices[:, i], ax=ax, linewidth=1.0, alpha=0.22)

        # Percentiles sobre el abanico
        p_low, p_med, p_high = np.percentile(prices, percentiles, axis=1)
        sns.lineplot(x=future_index, y=p_med, ax=ax, linewidth=2.2, label=f"Mediana (P{percentiles[1]})")
        ax.fill_between(future_index, p_low, p_high, alpha=0.18,
                        label=f"Banda P{percentiles[0]}–P{percentiles[2]}")

        # Último precio (marker)
        last_price = float(df["price"].iloc[-1])
        ax.axhline(last_price, linestyle=":", linewidth=1.2, alpha=0.6, label="Último precio")

        # Ejes / formato
        ax.set_title(title or f"Historical + Monte Carlo — {self.symbol}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(ylabel)
        if currency_symbol:
            ax.yaxis.set_major_formatter(_currency_fmt(currency_symbol))
        _format_dates(ax)
        ax.legend(frameon=False)
        fig.tight_layout()

        if path:
            _ensure_dir(path)
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        return fig, ax

    # -------- 4) Histograma terminal (seaborn) --------
    def plot_final_hist(
        self,
        prices: np.ndarray,
        bins: int = 50,
        figsize: tuple[int, int] = (9, 4.5),
        currency_symbol: Optional[str] = None,
        title: Optional[str] = None,
        dpi: int = 150,
        path: Optional[str] = None,
    ):
        if prices is None or prices.size == 0:
            return None, None
        finals = prices[-1, :] if prices.ndim == 2 else prices

        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(finals, bins=bins, ax=ax, kde=True, alpha=0.85)

        mu = float(np.mean(finals))
        med = float(np.median(finals))
        ax.axvline(mu, ls="--", lw=1.6, label=f"Media: {mu:,.2f}")
        ax.axvline(med, ls=":",  lw=1.6, label=f"Mediana: {med:,.2f}")
        ax.legend(frameon=False)

        ax.set_title(title or f"Terminal Value Distribution — {self.symbol}")
        ax.set_xlabel("Valor terminal")
        ax.set_ylabel("Frecuencia")
        if currency_symbol:
            ax.xaxis.set_major_formatter(_currency_fmt(currency_symbol))
        fig.tight_layout()

        if path:
            _ensure_dir(path)
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        return fig, ax

# ========== Clase Report opcional ==========
@dataclass
class MonteCarloReport(PlotMixinSeaborn):
    """
    Orquesta plots y permite exportar a un único PDF.
    """
    symbol: str
    price_series: pd.DataFrame | pd.Series | object

    def to_pdf(
        self,
        prices: np.ndarray,
        pdf_path: str,
        *,
        include_hist: bool = True,
        include_hist_sims: bool = True,
        include_final_hist: bool = True,
        dpi: int = 150,
    ):
        _ensure_dir(pdf_path)
        with PdfPages(pdf_path) as pdf:
            if include_hist:
                fig, _ = self.plot_history()
                if fig: 
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)
            if include_hist_sims:
                fig, _ = self.plot_history_with_simulations(prices=prices)
                if fig:
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)
            if include_final_hist:
                fig, _ = self.plot_final_hist(prices=prices)
                if fig:
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)
        return pdf_path
