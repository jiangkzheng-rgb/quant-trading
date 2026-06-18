#!/usr/bin/env python3
"""
多因子趋势信号系统 — 通用港美股买卖信号生成器
运行: python strategy.py
     python strategy.py AAPL NVDA 0700.HK TSLA
结果: reports/signals_data.js  (自动刷新网页)
"""

import sys, json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── 默认自选股（可随时修改） ──────────────────────────────────────────────────
DEFAULT_WATCHLIST = [
    # 美股宽基 ETF
    "QQQ", "SPY", "IWM",
    # 美股科技
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA",
    # 美股半导体
    "TSM", "AMD", "AVGO", "QCOM", "ASML",
    # 港股 (Yahoo Finance 格式)
    "0700.HK",   # 腾讯
    "9988.HK",   # 阿里巴巴
    "3690.HK",   # 美团
    "0941.HK",   # 中国移动
    "1810.HK",   # 小米
    "9999.HK",   # 网易
]

# ── 策略参数 ──────────────────────────────────────────────────────────────────
PARAMS = {
    "ema_fast":  20,
    "ema_mid":   50,
    "ema_slow":  200,
    "rsi_period": 14,
    "atr_period": 14,
    "macd_fast":  12,
    "macd_slow":  26,
    "macd_sig":    9,
    "vol_period": 20,
    "atr_stop":    2.0,   # 止损 = 入场价 - 2×ATR
    "atr_target":  4.0,   # 目标 = 入场价 + 4×ATR（2:1 盈亏比）
    "buy_threshold":  62,
    "sell_threshold": 38,
}


# ── 数据获取 ──────────────────────────────────────────────────────────────────
def get_data(ticker: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False, multi_level_index=False)
    if df.empty or len(df) < 60:
        return None
    return df


# ── 指标计算 ──────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    h = df["High"]
    lo = df["Low"]
    v = df["Volume"]

    # 均线
    df["ema20"]  = c.ewm(span=PARAMS["ema_fast"]).mean()
    df["ema50"]  = c.ewm(span=PARAMS["ema_mid"]).mean()
    df["ema200"] = c.ewm(span=PARAMS["ema_slow"]).mean()

    # MACD
    ema12 = c.ewm(span=PARAMS["macd_fast"]).mean()
    ema26 = c.ewm(span=PARAMS["macd_slow"]).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=PARAMS["macd_sig"]).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(PARAMS["rsi_period"]).mean()
    loss  = (-delta.clip(upper=0)).rolling(PARAMS["rsi_period"]).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # ATR
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(PARAMS["atr_period"]).mean()

    # 成交量比
    df["vol_ratio"] = v / v.rolling(PARAMS["vol_period"]).mean()

    # 布林带百分比
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_pct"] = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    # 近 5 日涨跌幅
    df["pct5"] = c.pct_change(5) * 100

    return df


# ── 信号评分 ──────────────────────────────────────────────────────────────────
def score_signal(df: pd.DataFrame, ticker: str) -> dict:
    r   = df.iloc[-1]   # 最新行
    r1  = df.iloc[-2]   # 前一行
    r5  = df.iloc[-5]   # 5日前

    close  = float(r["Close"])
    ema20  = float(r["ema20"])
    ema50  = float(r["ema50"])
    ema200 = float(r["ema200"])
    macd   = float(r["macd"])
    msig   = float(r["macd_sig"])
    mhist  = float(r["macd_hist"])
    mhist1 = float(r1["macd_hist"])
    rsi    = float(r["rsi"])
    atr    = float(r["atr"])
    vr     = float(r["vol_ratio"])
    bb     = float(r["bb_pct"])
    pct5   = float(r["pct5"])

    score  = 50   # 基准分 50
    factors = []  # (分数变化, 描述)

    # ── 趋势层（最高 ±30）──────────────────────────────
    if ema20 > ema50:
        score += 15; factors.append((+15, f"EMA20 {ema20:.2f} > EMA50 {ema50:.2f}，短期上行"))
    else:
        score -= 15; factors.append((-15, f"EMA20 {ema20:.2f} < EMA50 {ema50:.2f}，短期下行"))

    if ema50 > ema200:
        score += 15; factors.append((+15, f"EMA50 > EMA200，长期牛市"))
    else:
        score -= 15; factors.append((-15, f"EMA50 < EMA200，长期熊市"))

    # ── 动量层（最高 ±25）──────────────────────────────
    if macd > msig:
        score += 12; factors.append((+12, "MACD 金叉"))
    else:
        score -= 12; factors.append((-12, "MACD 死叉"))

    if mhist > mhist1:
        score += 13; factors.append((+13, "MACD 柱状线扩张，动量加速"))
    else:
        score -= 8;  factors.append((-8,  "MACD 柱状线收缩，动量减弱"))

    # ── RSI 层（最高 ±20）──────────────────────────────
    if 40 <= rsi <= 65:
        score += 15; factors.append((+15, f"RSI {rsi:.1f} 健康区间（40-65）"))
    elif rsi < 30:
        score += 10; factors.append((+10, f"RSI {rsi:.1f} 超卖，潜在反弹"))
    elif rsi >= 80:
        score -= 20; factors.append((-20, f"RSI {rsi:.1f} 严重超买，风险高"))
    elif rsi >= 70:
        score -= 10; factors.append((-10, f"RSI {rsi:.1f} 偏高，注意回调"))
    elif rsi < 40:
        score -= 5;  factors.append((-5,  f"RSI {rsi:.1f} 偏弱"))

    # ── 成交量层（最高 ±10）──────────────────────────────
    if vr >= 1.5:
        score += 10; factors.append((+10, f"成交量放大 {vr:.1f}×，强力确认"))
    elif vr >= 1.2:
        score += 5;  factors.append((+5,  f"成交量温和放大 {vr:.1f}×"))
    elif vr <= 0.6:
        score -= 8;  factors.append((-8,  f"成交量萎缩 {vr:.1f}×，信号弱"))

    # ── 布林带位置（最高 ±5）──────────────────────────────
    if 0.2 <= bb <= 0.8:
        score += 5;  factors.append((+5, f"布林带中位区域，空间充足"))
    elif bb > 0.95:
        score -= 5;  factors.append((-5, "触及布林带上轨，超买压力"))
    elif bb < 0.05:
        score += 3;  factors.append((+3, "触及布林带下轨，超卖反弹机会"))

    score = int(max(0, min(100, score)))

    # 判断信号
    if score >= PARAMS["buy_threshold"]:
        action = "BUY"
    elif score <= PARAMS["sell_threshold"]:
        action = "SELL"
    else:
        action = "HOLD"

    # 止损与目标
    stop   = round(close - PARAMS["atr_stop"]   * atr, 2)
    target = round(close + PARAMS["atr_target"]  * atr, 2)
    risk   = round(close - stop,  2)
    reward = round(target - close, 2)

    # 只取影响最大的三条因子说明
    factors_sorted = sorted(factors, key=lambda x: abs(x[0]), reverse=True)[:3]
    top_reasons    = [f[1] for f in factors_sorted]

    return {
        "ticker":      ticker,
        "action":      action,
        "score":       score,
        "close":       round(close, 2),
        "stop_loss":   stop,
        "take_profit": target,
        "risk":        risk,
        "reward":      reward,
        "atr":         round(atr, 2),
        "rsi":         round(rsi, 1),
        "macd":        round(macd, 4),
        "macd_sig":    round(msig, 4),
        "ema20":       round(ema20, 2),
        "ema50":       round(ema50, 2),
        "ema200":      round(ema200, 2),
        "vol_ratio":   round(vr, 2),
        "bb_pct":      round(bb, 2),
        "pct5":        round(pct5, 2),
        "reasons":     top_reasons,
        "updated":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ── 单只股票分析 ──────────────────────────────────────────────────────────────
def analyze(ticker: str) -> dict | None:
    label = {"BUY": "▲ 买入", "SELL": "▼ 卖出", "HOLD": "◆ 观望"}
    print(f"  {ticker:<12}", end="", flush=True)
    try:
        df = get_data(ticker)
        if df is None:
            print("数据不足，跳过")
            return None
        df = compute_indicators(df)
        sig = score_signal(df, ticker)
        print(f"${sig['close']:<8}  {label[sig['action']]}  评分 {sig['score']}")
        return sig
    except Exception as e:
        print(f"错误: {e}")
        return None


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    tickers = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_WATCHLIST
    print(f"\n{'═'*55}")
    print(f"  多因子趋势信号系统  —  {len(tickers)} 只标的")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*55}")

    results = [sig for t in tickers if (sig := analyze(t))]
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── 汇总打印 ──────────────────────────────────────────
    buys  = [r for r in results if r["action"] == "BUY"]
    sells = [r for r in results if r["action"] == "SELL"]
    holds = [r for r in results if r["action"] == "HOLD"]

    print(f"\n{'─'*55}")
    if buys:
        print(f"\n▲  买入信号 ({len(buys)})：")
        for r in buys:
            print(f"   {r['ticker']:<10} 评分 {r['score']:3}  入场 ${r['close']:<8} "
                  f"止损 ${r['stop_loss']:<8} 目标 ${r['take_profit']}")
    if sells:
        print(f"\n▼  卖出信号 ({len(sells)})：")
        for r in sells:
            print(f"   {r['ticker']:<10} 评分 {r['score']:3}  当前 ${r['close']}")
    if holds:
        print(f"\n◆  观望 ({len(holds)})：{', '.join(r['ticker'] for r in holds)}")

    # ── 写出 JS 数据文件（供网页自动加载）──────────────────
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(results),
        "signals": results,
    }
    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)

    js_path = out_dir / "signals_data.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(f"/* 自动生成，勿手动编辑 — {output['generated_at']} */\n")
        f.write(f"window.SIGNAL_DATA = {json.dumps(output, ensure_ascii=False, indent=2)};\n")

    print(f"\n✓  数据已写入 reports/signals_data.js，刷新网页即可查看最新信号")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
