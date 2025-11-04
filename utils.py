import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

def to_business_days(df: pd.Series, how: Literal["ffill", "bfill", "none"] = "ffill") -> pd.DataFrame:
    """
    Reindexa el df a días laborables (lunes-viernes) entre min(fecha) y max(fecha).
    - how: 'ffill' (por defecto), 'bfill', 'none' (no rellena).
    """

    if df.empty:
        return df

    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("El índice debe ser de fechas (DatetimeIndex/PeriodIndex).")
    
    
    start, end = df.index.min(), df.index.max()
    
    # Genera todos los días hábiles entre el start y end
    bidx = pd.date_range(start=start, end=end, freq='B') # 'B' = business days
    df2 = df.reindex(bidx) #inserta filas para los laborables que faltan
    if how == "ffill":
        df2 = df2.ffill()
    elif how == "bfill":
        df2 = df2.bfill()
    elif how == "none":
        pass
    else:
        raise ValueError("how debe ser 'ffill', 'bfill' o 'none'.")
    return df2


def clean_price_frame(df: pd.Series, price_col: str = "price") -> pd.DataFrame:
    """
    Limpieza del DataFrame de precios.
    Indica df y price_col; devuelve df limpio.
    
    1. Asegura índice temporal (DatetimeIndex, sin zona horaria, ordenado, único)
    2. Elimina valores NA / no finitos
    3. Elimina precios no positivos
    4. Elimina duplicados
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    # timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=[price_col])
    df = df[np.isfinite(df[price_col])]
    df = df[df[price_col] > 0]
    return df

def log_returns(price: pd.Series) -> pd.Series:
    """Log return diario natural; elimina primer NA."""
    return np.log(price).diff().dropna() #lo mismo que np.log(price / price.shift(1)). EL dropna elimina el primer NA

def annualize_stats(mu_daily: float, sigma_daily: float, trading_days: int = 252) -> Tuple[float, float]:
    """Transformnar estadísticos diarios a anuales."""
    mu_ann = mu_daily * trading_days
    sigma_ann = sigma_daily * np.sqrt(trading_days)
    return mu_ann, sigma_ann

def sharpe_ratio(mu_daily: float, sigma_daily: float, rf_daily: float = 0.0) -> float:
    """
    Ratio de Sharpe diari. Indica rendimiento extra por unidad de riesgo.
    rf_daily > 0 para tasa libre de riesgo.
    """
    if sigma_daily == 0:
        return np.nan
    return (mu_daily - rf_daily) / sigma_daily

def drawdowns(price: pd.Series) -> pd.Series:
    """Serie sobre drawdowns (pérdidas desde el máximo histórico)."""
    peak = price.cummax()
    dd = price / peak - 1.0
    return dd

def var_cvar(returns: pd.Series, alpha: float = 0.05) -> tuple:
    """
    Histórico VaR/CVaR al nivel alpha para una serie de retornos.
    VaR: Pérdida máxima esperada de una cartera durante un período dado con nivel confianza alpha
    CVaR: Pérdida media condicionada a que la pérdida exceda el VaR
    """
    if returns.empty:
        return (np.nan, np.nan)
    cutoff = np.quantile(returns, alpha)
    cvar = returns[returns <= cutoff].mean() if (returns <= cutoff).any() else np.nan
    return cutoff, cvar