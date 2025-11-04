import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from montecarlo import *
from models import *
from utils import *
from providers import *

sns.set_theme(context="notebook", style="whitegrid")

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _format_dates(ax):
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))

def _currency_fmt(symbol: str = "€"):
    return FuncFormatter(lambda y, _: f"{y:,.2f}{symbol}".replace(",", "X").replace(".", ",").replace("X", "."))


class MonteCarloPlots:
    def __init__(self, mc: MonteCarloSimulation):
        self.mc = mc

    def plot_history(self, path: str | None = None, currency_symbol: str | None = None):
        df = self.mc._price_df
        if df.empty:
            return None, None
        fig, ax = plt.subplots(figsize=(10,4))
        sns.lineplot(x=df.index, y=df["price"], ax=ax, lw=1.9, label="Histórico")
        ax.set_title(f"Historical Price — {self.mc.symbol}")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Precio")
        if currency_symbol: ax.yaxis.set_major_formatter(_currency_fmt(currency_symbol))
        _format_dates(ax); ax.legend(frameon=False); fig.tight_layout()
        if path: _ensure_dir(path); fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)
        return fig, ax

    def plot_simulations(self, prices: np.ndarray, max_paths: int = 50, index=None,
                         percentiles=(5,50,95), path: str | None = None):
        if prices is None or prices.size == 0: return None, None
        steps, n_sims = prices.shape
        k = min(max_paths, n_sims)
        x = np.arange(steps) if index is None else np.array(list(index))
        fig, ax = plt.subplots(figsize=(10,5))
        for i in range(k):
            sns.lineplot(x=x, y=prices[:, i], ax=ax, lw=1.0, alpha=0.25)
        p_low, p_med, p_high = np.percentile(prices, percentiles, axis=1)
        sns.lineplot(x=x, y=p_med, ax=ax, lw=2.2, label=f"P{percentiles[1]}")
        ax.fill_between(x, p_low, p_high, alpha=0.18, label=f"P{percentiles[0]}–P{percentiles[2]}")
        ax.set_title(f"Monte Carlo Simulations — {self.mc.symbol}")
        ax.set_xlabel("Fecha" if index is not None else "Paso"); ax.set_ylabel("Precio simulado")
        if index is not None: _format_dates(ax)
        ax.legend(frameon=False); fig.tight_layout()
        if path: _ensure_dir(path); fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)
        return fig, ax

    def plot_history_with_simulations(self, prices: np.ndarray, use_business_days=True,
                                      max_paths=50, percentiles=(5,50,95),
                                      currency_symbol: str | None = None,
                                      path: str | None = None):
        df = self.mc._price_df
        if df.empty or prices is None or prices.size == 0: return None, None
        last_dt = df.index[-1]
        steps, n_sims = prices.shape
        if use_business_days:
            future_index = pd.bdate_range(last_dt, periods=steps+1, inclusive="right")
        else:
            future_index = pd.date_range(last_dt, periods=steps+1, inclusive="right", freq="D")
        fig, ax = plt.subplots(figsize=(11,5.5))
        sns.lineplot(x=df.index, y=df["price"], ax=ax, lw=1.9, label="Histórico")
        k = min(max_paths, n_sims)
        for i in range(k):
            sns.lineplot(x=future_index, y=prices[:, i], ax=ax, lw=1.0, alpha=0.22)
        p_low, p_med, p_high = np.percentile(prices, percentiles, axis=1)
        sns.lineplot(x=future_index, y=p_med, ax=ax, lw=2.2, label=f"Mediana (P{percentiles[1]})")
        ax.fill_between(future_index, p_low, p_high, alpha=0.18, label=f"Banda P{percentiles[0]}–P{percentiles[2]}")
        last_price = float(df["price"].iloc[-1])
        ax.axhline(last_price, ls=":", lw=1.2, alpha=0.6, label="Último precio")
        ax.set_title(f"Historical + Monte Carlo — {self.mc.symbol}")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Precio")
        if currency_symbol: ax.yaxis.set_major_formatter(_currency_fmt(currency_symbol))
        _format_dates(ax); ax.legend(frameon=False); fig.tight_layout()
        if path: _ensure_dir(path); fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)
        return fig, ax

    def plot_final_hist(self, prices: np.ndarray, bins: int = 50,
                        currency_symbol: str | None = None, path: str | None = None):
        if prices is None or prices.size == 0: return None, None
        finals = prices[-1, :] if prices.ndim == 2 else prices
        fig, ax = plt.subplots(figsize=(9,4.5))
        sns.histplot(finals, bins=bins, ax=ax, kde=True, alpha=0.85)
        mu, med = float(np.mean(finals)), float(np.median(finals))
        ax.axvline(mu, ls="--", lw=1.6, label=f"Media: {mu:,.2f}")
        ax.axvline(med, ls=":",  lw=1.6, label=f"Mediana: {med:,.2f}")
        ax.set_title(f"Terminal Value Distribution — {self.mc.symbol}")
        ax.set_xlabel("Valor terminal"); ax.set_ylabel("Frecuencia")
        if currency_symbol: ax.xaxis.set_major_formatter(_currency_fmt(currency_symbol))
        ax.legend(frameon=False); fig.tight_layout()
        if path: _ensure_dir(path); fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)
        return fig, ax


class MonteCarloReport:
    """
    Informe visual para cualquier activo/cartera compatible con MonteCarloSimulation.
    """
    def __init__(self, mc: MonteCarloSimulation):
        self.mc = mc
        self.plots = MonteCarloPlots(mc)

    def to_pdf(self, prices: np.ndarray, pdf_path: str,
               include_hist=True, include_hist_sims=True, include_final_hist=True):
        _ensure_dir(pdf_path)
        with PdfPages(pdf_path) as pdf:
            if include_hist:
                fig, _ = self.plots.plot_history()
                if fig: pdf.savefig(fig, dpi=150); plt.close(fig)
            if include_hist_sims:
                fig, _ = self.plots.plot_history_with_simulations(prices=prices)
                if fig: pdf.savefig(fig, dpi=150); plt.close(fig)
            if include_final_hist:
                fig, _ = self.plots.plot_final_hist(prices=prices)
                if fig: pdf.savefig(fig, dpi=150); plt.close(fig)
        return pdf_path
