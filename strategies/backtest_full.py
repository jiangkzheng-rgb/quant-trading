"""
全量样本回测 — 多策略 × 多资产 × 多周期
覆盖：2010-01-01 至今（约15年）
策略：MA交叉 / RSI反转 / 布林带突破 / 动量
资产：QQQ / SPY / AAPL / MSFT / NVDA / TSM / 0700.HK / 9988.HK
"""
import warnings, sys, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt
import quantstats as qs
import yfinance as yf
from datetime import datetime

# ─── 配置 ────────────────────────────────────────────────
START      = "2010-01-01"
END        = datetime.today().strftime("%Y-%m-%d")
# 样本内/样本外分割
SPLIT_DATE = "2023-01-01"

TICKERS = ["QQQ", "SPY", "AAPL", "MSFT", "NVDA", "TSM", "0700.HK", "9988.HK"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*65}")
print(f"  全量样本回测  {START} → {END}  ({SPLIT_DATE} 前后分割)")
print(f"{'='*65}")

# ─── 1. 下载数据 ─────────────────────────────────────────
print("\n[1/5] 下载历史数据...")
raw = yf.download(TICKERS, start=START, end=END,
                  auto_adjust=True, progress=True)
price = raw["Close"].dropna(how="all")

# 港股数据往往缺失较多，单独处理
available = [t for t in TICKERS if price[t].notna().sum() > 200]
price = price[available]
print(f"  有效标的: {available}")
print(f"  数据行数: {len(price)}  ({price.index[0].date()} → {price.index[-1].date()})")

# 样本内 / 样本外
price_in  = price[price.index < SPLIT_DATE]
price_out = price[price.index >= SPLIT_DATE]

# ─── 工具函数 ─────────────────────────────────────────────
def run_portfolio(price_df, entries_df, exits_df, label=""):
    pf = vbt.Portfolio.from_signals(
        price_df, entries_df, exits_df,
        freq="D", fees=0.001, slippage=0.001  # 0.1% 手续费+滑点
    )
    return pf

def fmt_stats(pf, label):
    s = pf.stats()
    ret    = s.get("Total Return [%]", float("nan"))
    sharpe = s.get("Sharpe Ratio",     float("nan"))
    mdd    = s.get("Max Drawdown [%]", float("nan"))
    wins   = s.get("Win Rate [%]",     float("nan"))
    trades = s.get("Total Trades",     float("nan"))

    # 尝试计算年化收益（CAGR via quantstats 或手动计算）
    try:
        total_days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
        years = total_days / 365.25
        total_r = float(ret) / 100
        ann = ((1 + total_r) ** (1 / years) - 1) * 100 if years > 0 and total_r > -1 else float("nan")
    except Exception:
        ann = float("nan")

    return {
        "策略": label,
        "年化收益%": round(ann, 1) if not np.isnan(ann) else "-",
        "总收益%":   round(float(ret), 1) if not np.isnan(float(ret)) else "-",
        "夏普比":    round(float(sharpe), 2) if not np.isnan(float(sharpe)) else "-",
        "最大回撤%": round(float(mdd), 1) if not np.isnan(float(mdd)) else "-",
        "胜率%":     round(float(wins), 1) if not np.isnan(float(wins)) else "-",
        "交易次数":  int(trades) if not np.isnan(float(trades)) else "-",
    }

all_results = []

# ─── 2. 策略一：双均线交叉（参数网格扫描）─────────────────
print("\n[2/5] 策略一：双均线交叉（参数网格扫描）")

MA_COMBOS = [(10, 30), (20, 50), (20, 100), (50, 200)]
ma_summary = []

for fast, slow in MA_COMBOS:
    for phase, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_clean = p.ffill().dropna(axis=1, thresh=int(len(p)*0.8))
            fast_ma = vbt.MA.run(p_clean, fast)
            slow_ma = vbt.MA.run(p_clean, slow)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits   = fast_ma.ma_crossed_below(slow_ma)
            pf = run_portfolio(p_clean, entries, exits)
            label = f"MA({fast}/{slow}) {phase}"
            ma_summary.append(fmt_stats(pf, label))
        except Exception as e:
            ma_summary.append({"策略": f"MA({fast}/{slow}) {phase}", "error": str(e)})

df_ma = pd.DataFrame(ma_summary)
print(df_ma.to_string(index=False))
all_results.extend(ma_summary)

# ─── 3. 策略二：RSI 超买超卖反转 ─────────────────────────
print("\n[3/5] 策略二：RSI 反转（多参数）")

RSI_PARAMS = [
    (14, 30, 70),   # 标准参数
    (7,  25, 75),   # 短周期激进
    (21, 35, 65),   # 长周期保守
]
rsi_summary = []

for period, lo, hi in RSI_PARAMS:
    for phase, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_clean = p.ffill().dropna(axis=1, thresh=int(len(p)*0.8))
            rsi = vbt.RSI.run(p_clean, period)
            entries = rsi.rsi_below(lo)
            exits   = rsi.rsi_above(hi)
            pf = run_portfolio(p_clean, entries, exits)
            label = f"RSI({period},{lo}/{hi}) {phase}"
            rsi_summary.append(fmt_stats(pf, label))
        except Exception as e:
            rsi_summary.append({"策略": f"RSI({period},{lo}/{hi}) {phase}", "error": str(e)})

df_rsi = pd.DataFrame(rsi_summary)
print(df_rsi.to_string(index=False))
all_results.extend(rsi_summary)

# ─── 4. 策略三：布林带突破 ───────────────────────────────
print("\n[4/5] 策略三：布林带突破")

BB_PARAMS = [(20, 2.0), (20, 1.5), (30, 2.0)]
bb_summary = []

for window, std in BB_PARAMS:
    for phase, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_clean = p.ffill().dropna(axis=1, thresh=int(len(p)*0.8))
            bb = vbt.BBANDS.run(p_clean, window, alpha=std)
            # 对齐 columns：bb.lower/upper 可能含 MultiIndex，提取 values
            lower = pd.DataFrame(bb.lower.values, index=p_clean.index, columns=p_clean.columns)
            upper = pd.DataFrame(bb.upper.values, index=p_clean.index, columns=p_clean.columns)
            entries = p_clean < lower
            exits   = p_clean > upper
            pf = run_portfolio(p_clean, entries, exits)
            label = f"BB({window},{std}) {phase}"
            bb_summary.append(fmt_stats(pf, label))
        except Exception as e:
            bb_summary.append({"策略": f"BB({window},{std}) {phase}", "error": str(e)})

df_bb = pd.DataFrame(bb_summary)
print(df_bb.to_string(index=False))
all_results.extend(bb_summary)

# ─── 5. 生成 quantstats 报告（QQQ 全周期）───────────────
print("\n[5/5] 生成 QuantStats 完整报告 (QQQ vs SPY, 2010-今)...")
try:
    qqq_rets = price["QQQ"].pct_change().dropna() if "QQQ" in price.columns else None
    spy_rets = price["SPY"].pct_change().dropna() if "SPY" in price.columns else None

    if qqq_rets is not None:
        html_path = os.path.join(OUT_DIR, "QQQ_full_report.html")
        qs.reports.html(
            qqq_rets,
            benchmark=spy_rets,
            output=html_path,
            title="QQQ Buy & Hold 2010-2024 vs SPY",
            download_filename="QQQ_full_report"
        )
        print(f"  HTML 报告: {html_path}")

        # 各个指标打印
        print(f"\n  QQQ 全周期统计 ({START} → {END})")
        print(f"  年化收益  : {qs.stats.cagr(qqq_rets)*100:.1f}%")
        print(f"  累计收益  : {qs.stats.comp(qqq_rets)*100:.0f}%")
        print(f"  夏普比率  : {qs.stats.sharpe(qqq_rets):.2f}")
        print(f"  卡玛比率  : {qs.stats.calmar(qqq_rets):.2f}")
        print(f"  最大回撤  : {qs.stats.max_drawdown(qqq_rets)*100:.1f}%")
        print(f"  波动率    : {qs.stats.volatility(qqq_rets)*100:.1f}%")
        print(f"  胜率(月)  : {qs.stats.win_rate(qqq_rets, aggregate='M')*100:.1f}%")
        print(f"  Sortino   : {qs.stats.sortino(qqq_rets):.2f}")
except Exception as e:
    print(f"  QuantStats 报告生成出错: {e}")

# ─── 综合汇总 ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("  综合汇总表（仅有效结果）")
print(f"{'='*65}")
valid = [r for r in all_results if "error" not in r and r.get("夏普比") != "-"]
if valid:
    df_all = pd.DataFrame(valid)
    # 按样本内/外分开展示，并排序
    for phase in ["样本内", "样本外"]:
        sub = df_all[df_all["策略"].str.contains(phase)].sort_values(
            "夏普比", ascending=False
        )
        print(f"\n  [{phase}] Top 策略:")
        print(sub.head(8).to_string(index=False))

print(f"\n  报告已保存至: {OUT_DIR}")
print(f"{'='*65}\n")
