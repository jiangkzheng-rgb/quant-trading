#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


OUT_DIR = Path(__file__).resolve().parent


@dataclass
class StrategyConfig:
    symbol: str
    name: str
    budget: float
    daily: float
    months: int
    reserve_pct: float
    max_extra: float
    asset_kind: str


CONFIGS = [
    StrategyConfig("^NDX", "nasdaq100", 500_000, 300, 18, 20, 25_000, "nasdaq"),
    StrategyConfig("GC=F", "gold", 50_000, 400, 12, 30, 5_000, "gold"),
]


def download_close(symbol: str) -> pd.Series:
    data = yf.download(symbol, period="max", auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"{symbol} has no data")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = symbol
    return close


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def drawdown_percentile(close: pd.Series, lookback: int) -> pd.Series:
    rolling_high = close.rolling(lookback, min_periods=60).max()
    dd_abs = (close / rolling_high - 1).abs()
    return dd_abs.rolling(lookback, min_periods=120).rank(pct=True) * 100


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def score_series(close: pd.Series, cfg: StrategyConfig) -> pd.Series:
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma200 = close.rolling(200).mean()
    ma200_slope = ma200 / ma200.shift(21) - 1
    dist200 = close / ma200 - 1
    one_year = close / close.shift(252) - 1
    dd_756 = close / close.rolling(756, min_periods=200).max() - 1
    dd_252 = close / close.rolling(252, min_periods=80).max() - 1
    dd_pct_756 = drawdown_percentile(close, 756)
    dd_pct_252 = drawdown_percentile(close, 252)
    rsi14 = rsi(close)
    vol60 = close.pct_change().rolling(60).std() * np.sqrt(252)

    score = pd.Series(10 if cfg.asset_kind == "nasdaq" else 8, index=close.index, dtype=float)

    if cfg.asset_kind == "nasdaq":
        score += np.select(
            [
                (dd_pct_756 >= 95) | (dd_756 <= -0.35),
                (dd_pct_756 >= 90) | (dd_756 <= -0.30),
                (dd_pct_756 >= 80) | (dd_756 <= -0.25),
                (dd_pct_756 >= 65) | (dd_756 <= -0.20),
                (dd_pct_756 >= 50) | (dd_756 <= -0.15),
                dd_756 <= -0.10,
            ],
            [58, 50, 42, 34, 24, 14],
            default=0,
        )
        score += np.select(
            [dist200 <= -0.15, dist200 <= -0.08, dist200 <= 0, (dist200 >= 0.18) & (dd_756 > -0.10), (dist200 >= 0.12) & (dd_756 > -0.08)],
            [18, 12, 6, -22, -14],
            default=0,
        )
        score += np.select([rsi14 <= 30, (rsi14 >= 72) & (dd_756 > -0.12)], [8, -8], default=0)
        score += np.where((ma200_slope < -0.03) & (dd_756 > -0.20), -6, 0)
        score += np.where((one_year >= 0.35) & (dd_756 > -0.10), -14, 0)
        score += np.where((vol60 >= 0.38) & (score >= 70), -5, 0)
    else:
        primary = pd.concat([dd_252, close / close.rolling(252, min_periods=80).max() - 1], axis=1).min(axis=1)
        score += np.select(
            [
                (dd_pct_252 >= 95) | (primary <= -0.18),
                (dd_pct_252 >= 88) | (primary <= -0.14),
                (dd_pct_252 >= 76) | (primary <= -0.10),
                (dd_pct_252 >= 62) | (primary <= -0.07),
                primary <= -0.04,
            ],
            [50, 42, 32, 22, 12],
            default=0,
        )
        score += np.where(close <= ma60, 4, 0)
        score += np.where(close <= ma120, 5, 0)
        score += np.select([(dist200 <= -0.06), (dist200 >= 0.10) & (one_year >= 0.20)], [7, -18], default=0)
        score += np.select([rsi14 <= 32, rsi14 >= 72], [5, -6], default=0)

    return score.clip(0, 100).fillna(0)


def simulate(close: pd.Series, cfg: StrategyConfig, use_strategy: bool) -> pd.Series:
    score = score_series(close, cfg)
    deployable = cfg.budget * (1 - cfg.reserve_pct / 100)
    days = max(1, cfg.months * 21)
    base_daily = min(cfg.daily, deployable / days)
    cash = cfg.budget
    units = 0.0
    equity = []

    for date, price in close.items():
      if cash <= 0:
          equity.append(units * price)
          continue
      buy = min(base_daily, cash)
      if use_strategy:
          s = score.loc[date]
          if cfg.asset_kind == "nasdaq":
              min_score = 58
              strength = max(0, (s - min_score) / (100 - min_score))
              extra = 0 if s < min_score else max(base_daily, cfg.daily) * (1.8 + 7.2 * strength) * 0.85
          else:
              min_score = 62
              strength = max(0, (s - min_score) / (100 - min_score))
              extra = 0 if s < min_score else max(base_daily, cfg.daily) * (1.2 + 4.5 * strength) * 0.85
          buy += min(extra, cfg.max_extra)
      buy = min(buy, cash)
      units += buy / price
      cash -= buy
      equity.append(units * price + cash)

    return pd.Series(equity, index=close.index, name="equity")


def summarize(close: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    start_dates = close.resample("MS").first().dropna().index
    rows = []
    for start in start_dates:
        sample = close.loc[start:].iloc[: cfg.months * 21 + 252]
        if len(sample) < cfg.months * 21:
            continue
        dca = simulate(sample, cfg, False)
        strategy = simulate(sample, cfg, True)
        rows.append(
            {
                "start": start.date().isoformat(),
                "dca_return": dca.iloc[-1] / cfg.budget - 1,
                "strategy_return": strategy.iloc[-1] / cfg.budget - 1,
                "excess": strategy.iloc[-1] / dca.iloc[-1] - 1,
                "dca_mdd": max_drawdown(dca),
                "strategy_mdd": max_drawdown(strategy),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    lines = ["# 量化库版加仓策略回测", ""]
    for cfg in CONFIGS:
        close = download_close(cfg.symbol).last("15Y")
        table = summarize(close, cfg)
        out_csv = OUT_DIR / f"quant_backtest_{cfg.name}.csv"
        table.to_csv(out_csv, index=False, encoding="utf-8-sig")

        lines.extend(
            [
                f"## {cfg.name}",
                "",
                f"- 样本数：{len(table)}",
                f"- 策略平均收益：{table['strategy_return'].mean():.2%}",
                f"- 等额定投平均收益：{table['dca_return'].mean():.2%}",
                f"- 平均超额：{table['excess'].mean():.2%}",
                f"- 跑赢比例：{(table['excess'] > 0).mean():.2%}",
                f"- 策略平均最大回撤：{table['strategy_mdd'].mean():.2%}",
                f"- 等额定投平均最大回撤：{table['dca_mdd'].mean():.2%}",
                f"- 明细：`{out_csv.name}`",
                "",
            ]
        )

    (OUT_DIR / "量化库版加仓策略回测.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
