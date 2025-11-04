from __future__ import annotations
import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


from utils import (
    clean_price_frame,
    to_business_days,
    annualize_stats,
    log_returns,
    sharpe_ratio,
    drawdowns,
    var_cvar,
)

@dataclass
class PriceSeries:
    symbol: str
    asset_type: str = "equity"  # 'equity' | 'index' | 'crypto' | 'fund'
    currency: str = "USD"
    provider: str = "unknown"
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["price"])) #Dataframe con columna 'price' vacía si no se proporciona

    # Estadísiticos básicos media y desviación estándar de los retornos logarítmicos diarios
    mu: float = field(init=False, default=np.nan)
    sigma: float = field(init=False, default=np.nan)

    def __post_init__(self): #Limpieza automática nada más entrar
        self.data = clean_price_frame(self.data)
        # Normaliza a business days con ffill
        self.data = to_business_days(self.data, how="ffill")
        # Calcula ahora los estadísticos básicos
        rets = log_returns(self.data["price"])
        if not rets.empty:
            self.mu = rets.mean()
            self.sigma = rets.std(ddof=1)

    # ---------- Constructores ----------
    @classmethod
    def from_dataframe(cls, symbol: str, df: pd.DataFrame, price_col: str = "price", **meta):
        """Crea PriceSeries desde DataFrame dado. Modifica el nombre de la columna de precio si es necesario."""
        if price_col not in df.columns:
            raise ValueError(f"Se esperaba '{price_col}' en df.")
        df2 = df[[price_col]].rename(columns={price_col: "price"})
        return cls(symbol=symbol, data=df2, **meta)

    # ---------- Core methods ----------
    def clean(self, method: str = "ffill") -> "PriceSeries":
        """Limpieza y reindex a business days por si metemos más datos después."""
        self.data = clean_price_frame(self.data)
        self.data = to_business_days(self.data, how=method)
        return self

    def resample(self, freq: str = "B") -> "PriceSeries":
        """Resample precios a otra frecuencia:'W', 'M'."""
        if self.data.empty:
            return self
        if freq == "B":
            return self
        # Usamos el último día del período como precio representativo
        self.data = self.data.resample(freq).last().dropna()
        # Volvemos a sacar los estadísticos básicos
        rets = self.log_returns()
        if not rets.empty:
            self.mu = rets.mean()
            self.sigma = rets.std(ddof=1)
        return self

    def log_returns(self) -> pd.Series:
        if self.data.empty:
            return pd.Series(dtype=float)
        return log_returns(self.data["price"])

    def extra_stats(self) -> Dict[str, float]:
        """Análisis más completo de estadísticos"""
        r = self.log_returns()
        if r.empty:
            return {k: np.nan for k in ["skew", "kurtosis", "sharpe_daily", "mu_ann", "sigma_ann", "var_95", "cvar_95"]}
        mu_daily = r.mean()
        sigma_daily = r.std(ddof=1)
        mu_ann, sigma_ann = annualize_stats(mu_daily, sigma_daily)
        s = {
            "skew": float(skew(r)),
            "kurtosis": float(kurtosis(r, fisher=True)),  # excess
            "sharpe_daily": float(sharpe_ratio(mu_daily, sigma_daily)),
            "mu_ann": float(mu_ann),
            "sigma_ann": float(sigma_ann),
        }
        v, c = var_cvar(r, alpha=0.05)
        s["var_95"] = float(v)
        s["cvar_95"] = float(c)
        return s

    # ---------- Monte Carlo ----------
    """"
    def monte_carlo(
        self,
        days: int = 252,
        n_sims: int = 2000,
        seed: int | None = 123,
        start_price: float | None = None,
    ) -> np.ndarray:
        Simula trayectorias futuras de precio usando un proceso tipo GBM.
        Usa mu y sigma empíricos calculados a partir de los log-returns
        históricos de esta serie.

        Devuelve un array de forma (days+1, n_sims).
       
        # 1) Comprobar que hay datos
        if self.data.empty:
            raise ValueError("No data in PriceSeries.")

        # 2) Precio inicial = último precio real o start_price
        price0 = float(self.data["price"].iloc[-1]) if start_price is None else float(start_price)

        # 3) Log-returns históricos
        r = self.log_returns()
        if r.empty:
            raise ValueError("Insufficient history for returns.")

        # 4) Estimar mu y sigma a partir de los datos
        mu = r.mean()
        sigma = r.std(ddof=1)

        if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
            raise ValueError("Invalid drift/vol estimates from history.")

        # 5) Generar matriz aleatoria N(0,1) de tamaño (days, n_sims)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((days, n_sims))

        # 6) Incrementos de GBM en log-precios. Genera una matriz (days, n_sims) con los incrementos diarios
        increments = (mu - 0.5 * sigma**2) + sigma * z

        # 7) Acumular en el tiempo (añadimos día 0). Genera matriz de log-precios en todas las simulaciones
        paths = np.vstack([ #esto solo pega la fila de ceros al principio
            np.zeros((1, n_sims)), #añade fila de ceros para el día 0 desplazamiento cero
            np.cumsum(increments, axis=0) #lo importante: suma acumulada de los logs 
        ])

        # 8) Pasar de log a precio
        prices = price0 * np.exp(paths)
        return prices


    def final_value_montecarlo(self, prices: np.ndarray) -> np.ndarray:
        Vector con últimos precios de cada simulación.
        return prices[-1, :]

    # ---------- Plot helpers ----------
    def plot_history(self, path: Optional[str] = None):
        if self.data.empty:
            return
        plt.figure()
        self.data["price"].plot(title=f"Historical Price - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()

    def plot_simulations(self, paths: np.ndarray, max_paths: int = 50, path: Optional[str] = None):
        k = min(max_paths, paths.shape[1])
        plt.figure()
        for i in range(k):
            plt.plot(paths[:, i])
        plt.title(f"Monte Carlo Simulations - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()

    def plot_final_hist(self, finals: np.ndarray, bins: int = 50, path: Optional[str] = None):
        plt.figure()
        plt.hist(finals, bins=bins)
        plt.title(f"Terminal Value Distribution - {self.symbol}")
        if path:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
    """
@dataclass
class Portfolio:
    positions: List[PriceSeries]
    weights: List[float]  # si no suma uno, se normalizan automáticamente
    name: str = "Cartera"
    currency: str = "USD"

    def __post_init__(self):
        if len(self.positions) != len(self.weights):
            raise ValueError("Se debe introducir el mismo número de posiciones y pesos.")
        w = np.array(self.weights, dtype=float)
        total = w.sum()
        if not np.isclose(total, 1.0):
            # Normalize to avoid errors; warn in report later
            self.weights = (w / total).tolist()

    # ---------- Core methods ----------
    def aligned_prices(self) -> pd.DataFrame:
        """Devuelve series de precios en la intersección de las fehcas"""
        frames = []
        for ps in self.positions:
            s = ps.data["price"].rename(ps.symbol)
            frames.append(s)
        df = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
        return df

    def value_series(self, initial_capital: float = 1.0) -> pd.Series:
        """Calcula la serie temporal del valor de la cartera."""
        df = self.aligned_prices()
        if df.empty:
            return pd.Series(dtype=float)
        w = np.array(self.weights)
        # Convertimos precios a log-returns
        rets = np.log(df).diff().dropna()
        # Weighted log-return
        port_log_ret = rets.dot(w)
        # Convert back to price-like equity curve
        eq = np.exp(port_log_ret.cumsum())
        # scale to initial_capital
        eq = initial_capital * eq / eq.iloc[0]
        eq.name = self.name
        return eq

    def log_returns(self) -> pd.Series:
        eq = self.value_series()
        return np.log(eq).diff().dropna()

    # ---------- Monte Carlo at portfolio level ----------
    def monte_carlo(
        self,
        days: int = 252,
        n_sims: int = 2000,
        seed: Optional[int] = 123,
        by_components: bool = False,
    ) -> np.ndarray:
        """
        Simulate future paths:
        - If by_components=False: simulate the portfolio using its aggregate mu/sigma.
        - If by_components=True: simulate each asset and combine with weights daily.
        Returns (days+1, n_sims) portfolio equity paths.
        """
        rng = np.random.default_rng(seed)
        if by_components:
            # Simulate each component independently (no correlation for simplicity)
            comp_paths = []
            for ps in self.positions:
                r = ps.log_returns()
                mu, sigma = r.mean(), r.std(ddof=1)
                z = rng.standard_normal((days, n_sims))
                inc = (mu - 0.5 * sigma**2) + sigma * z
                s0 = ps.data["price"].iloc[-1]
                paths = s0 * np.exp(np.vstack([np.zeros((1, n_sims)), np.cumsum(inc, axis=0)]))
                comp_paths.append(paths)
            # Combine by weights by converting to daily returns and weighting
            w = np.array(self.weights)[:, None]  # shape (k,1)
            # Compute portfolio path from weighted sum of normalized returns
            # Convert component prices to log-returns per day then aggregate
            agg = None
            for i, paths in enumerate(comp_paths):
                # Compute log returns from simulated prices
                lr = np.diff(np.log(paths), axis=0)  # shape (days, sims)
                contrib = w[i] * lr
                agg = contrib if agg is None else agg + contrib
            # Convert back to equity curve
            eq_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(agg, axis=0)])
            # Start at 1.0
            eq_paths = np.exp(eq_paths)
            return eq_paths
        else:
            # Aggregate stats from portfolio historical returns
            r = self.log_returns()
            if r.empty:
                raise ValueError("Portfolio has insufficient history.")
            mu, sigma = r.mean(), r.std(ddof=1)
            z = rng.standard_normal((days, n_sims))
            inc = (mu - 0.5 * sigma**2) + sigma * z
            paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(inc, axis=0)])
            eq_paths = np.exp(paths)  # start at 1.0
            return eq_paths

    # ---------- Reporting ----------
    def report(self, include_warnings: bool = True) -> str:
        """
        Generate a Markdown report with stats, warnings, risks, etc.
        """
        lines = []
        lines.append(f"# Portfolio Report — {self.name}")
        lines.append(f"- Currency: **{self.currency}**")
        # Composition
        lines.append("## Composition")
        for ps, w in zip(self.positions, self.weights):
            lines.append(f"- {ps.symbol} ({ps.asset_type}, {ps.provider}) — weight: **{w:.2%}**")

        # Price alignment
        df = self.aligned_prices()
        if df.empty:
            lines.append("\n> ⚠️ Not enough overlapping data to compute stats.")
            return "\n".join(lines)

        # Portfolio-level stats
        r = self.log_returns()
        if r.empty:
            lines.append("\n> ⚠️ Not enough returns to compute stats.")
            return "\n".join(lines)

        mu, sigma = r.mean(), r.std(ddof=1)
        mu_ann, sigma_ann = (mu * 252, sigma * np.sqrt(252))
        sr = (mu / sigma) if sigma > 0 else np.nan
        v95, c95 = var_cvar(r, alpha=0.05)

        lines.append("\n## Portfolio Stats (daily)")
        lines.append(f"- Mean (μ): **{mu:.6f}**")
        lines.append(f"- Std (σ): **{sigma:.6f}**")
        lines.append(f"- Sharpe (daily): **{sr:.3f}**")
        lines.append(f"- Annualized μ: **{mu_ann:.3f}**, Annualized σ: **{sigma_ann:.3f}**")
        lines.append(f"- VaR 95% (daily): **{v95:.4f}**, CVaR 95%: **{c95:.4f}**")

        # Component stats
        lines.append("\n## Components (daily)")
        for ps in self.positions:
            stats = ps.extra_stats()
            lines.append(f"### {ps.symbol}")
            lines.append(f"- μ: **{ps.mu:.6f}**, σ: **{ps.sigma:.6f}**, Sharpe(d): **{stats['sharpe_daily']:.3f}**")
            lines.append(f"- Skew: **{stats['skew']:.3f}**, Kurtosis(excess): **{stats['kurtosis']:.3f}**")
            lines.append(f"- VaR95: **{stats['var_95']:.4f}**, CVaR95: **{stats['cvar_95']:.4f}**")
            lines.append("")

        # Warnings
        if include_warnings:
            if not np.isclose(sum(self.weights), 1.0):
                lines.append("> ⚠️ Weights were auto-normalized to sum to 1.")
            if (df <= 0).any().any():
                lines.append("> ⚠️ Non-positive prices detected and filtered.")
            if df.isna().any().any():
                lines.append("> ⚠️ Missing values were forward-filled; results may be sensitive.")
        return "\n".join(lines)

    def plots_report(
        self,
        outdir: str = "./outputs",
        max_paths: int = 50,
        n_sims: int = 1000,
        days: int = 252,
        seed: int = 7
    ) -> Dict[str, str]:
        """
        Generate and save a set of useful visualizations. Returns paths to files.
        - Portfolio historical equity curve
        - Drawdown curve
        - Monte Carlo simulated paths (portfolio)
        - Histogram of terminal values (portfolio)
        - Historical price per asset
        """
        os.makedirs(outdir, exist_ok=True)
        paths = {}

        # Portfolio history
        eq = self.value_series(initial_capital=1.0)
        if not eq.empty:
            plt.figure()
            eq.plot(title=f"Portfolio Equity - {self.name}")
            p = os.path.join(outdir, f"{self.name}_equity.png")
            plt.savefig(p, bbox_inches="tight"); plt.close()
            paths["portfolio_equity"] = p

            # Drawdowns
            plt.figure()
            drawdowns(eq).plot(title=f"Portfolio Drawdown - {self.name}")
            p = os.path.join(outdir, f"{self.name}_drawdown.png")
            plt.savefig(p, bbox_inches="tight"); plt.close()
            paths["portfolio_drawdown"] = p

        # Portfolio Monte Carlo (aggregate method)
        mc_paths = self.monte_carlo(days=days, n_sims=n_sims, seed=seed, by_components=False)
        plt.figure()
        k = min(max_paths, mc_paths.shape[1])
        for i in range(k):
            plt.plot(mc_paths[:, i])
        plt.title(f"Monte Carlo (Aggregate) - {self.name}")
        p = os.path.join(outdir, f"{self.name}_mc_paths.png")
        plt.savefig(p, bbox_inches="tight"); plt.close()
        paths["portfolio_mc_paths"] = p

        finals = mc_paths[-1, :]
        plt.figure()
        plt.hist(finals, bins=50)
        plt.title(f"Terminal Value Dist (Aggregate) - {self.name}")
        p = os.path.join(outdir, f"{self.name}_mc_term_hist.png")
        plt.savefig(p, bbox_inches="tight"); plt.close()
        paths["portfolio_mc_term_hist"] = p

        # Component price histories
        for ps in self.positions:
            pth = os.path.join(outdir, f"{ps.symbol}_history.png")
            ps.plot_history(path=pth)
            paths[f"{ps.symbol}_history"] = pth

        return paths

