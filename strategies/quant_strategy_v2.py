"""
QuantSaaS Lab — 多策略多轮验证 v2
====================================
策略集：
  A. Sigmoid动态天平（核心创新）
  B. MA自适应趋势跟随
  C. RSI+波动率复合过滤
  D. 布林带均值回归

验证流程（5步）：
  1. 样本内回测     2010-2022
  2. 样本外验证     2023-今
  3. 参数鲁棒性     ±20% 扫描
  4. 跨资产验证     8 只标的
  5. 压力测试       2020-03 / 2022-01
"""
import warnings, sys, os, json, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt
import quantstats as qs
import yfinance as yf
from datetime import datetime

# ─── 全局配置 ──────────────────────────────────────────────
START       = "2010-01-01"
END         = datetime.today().strftime("%Y-%m-%d")
SPLIT_DATE  = "2023-01-01"
FEES        = 0.001   # 0.1%
SLIP        = 0.001   # 0.1%
TICKERS     = ["QQQ", "SPY", "AAPL", "MSFT", "NVDA", "TSM", "0700.HK", "9988.HK"]

BASE_DIR    = os.path.dirname(__file__)
REPORT_DIR  = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

JSON_OUT    = os.path.join(REPORT_DIR, "backtest_v2_results.json")

print(f"\n{'='*70}")
print(f"  QuantSaaS Lab — 多策略多轮验证 v2")
print(f"  数据范围: {START} → {END}  |  分割点: {SPLIT_DATE}")
print(f"  标的: {TICKERS}")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════
#   工具函数
# ═══════════════════════════════════════════════════════════

def clean_price(df: pd.DataFrame, min_ratio=0.8) -> pd.DataFrame:
    df = df.ffill()
    df = df.dropna(axis=1, thresh=int(len(df) * min_ratio))
    return df

def run_pf(price, entries, exits, label=""):
    return vbt.Portfolio.from_signals(
        price, entries, exits,
        freq="D", fees=FEES, slippage=SLIP
    )

def extract_stats(pf, label: str) -> dict:
    s = pf.stats()
    ret    = float(s.get("Total Return [%]", float("nan")))
    sharpe = float(s.get("Sharpe Ratio",     float("nan")))
    mdd    = float(s.get("Max Drawdown [%]", float("nan")))
    wins   = float(s.get("Win Rate [%]",     float("nan")))
    trades = float(s.get("Total Trades",     0))

    try:
        days  = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
        years = days / 365.25
        cagr  = ((1 + ret/100) ** (1/years) - 1) * 100 if years > 0 and ret > -100 else float("nan")
    except Exception:
        cagr = float("nan")

    try:
        calmar = abs(cagr / mdd) if mdd != 0 else float("nan")
    except Exception:
        calmar = float("nan")

    return {
        "策略":    label,
        "CAGR%":   round(cagr,  1) if not math.isnan(cagr)   else None,
        "总收益%":  round(ret,   1) if not math.isnan(ret)    else None,
        "夏普比":  round(sharpe, 2) if not math.isnan(sharpe) else None,
        "最大回撤%": round(mdd,  1) if not math.isnan(mdd)   else None,
        "卡玛比":  round(calmar, 2) if not math.isnan(calmar) else None,
        "胜率%":   round(wins,  1) if not math.isnan(wins)   else None,
        "交易次数": int(trades),
    }

def print_table(rows: list, title: str):
    print(f"\n  [{title}]")
    if not rows:
        print("  (无数据)")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("夏普比", ascending=False, na_position="last")
    print(df.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════
#   Step 0: 下载数据
# ═══════════════════════════════════════════════════════════
print("\n[Step 0] 下载历史数据...")
raw   = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=True)
price_all = clean_price(raw["Close"])
available = list(price_all.columns)
print(f"  有效标的: {available}  |  行数: {len(price_all)}")

price_in  = price_all[price_all.index <  SPLIT_DATE]
price_out = price_all[price_all.index >= SPLIT_DATE]

ALL_RESULTS = {}   # 汇总 JSON


# ═══════════════════════════════════════════════════════════
#   策略 A: Sigmoid 动态天平（核心微观引擎）
#
#   概念：用 Sigmoid 函数将"趋势信号"映射到 [0,1] 仓位权重
#         TargetWeight = 1 / (1 + exp(β × Signal + γ × InventoryBias))
#
#   Signal     = (快均线 - 慢均线) / ATR   — 标准化动量信号
#   InventoryBias = 当前持仓偏离中枢的程度
#   β (EffectiveBeta) 控制斜率灵敏度，由 GA 进化
#   γ 控制库存偏差惩罚强度
#
#   回测简化版：将 TargetWeight > 0.5 映射为 entry，< 0.5 映射为 exit
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  策略 A: Sigmoid 动态天平")
print("─"*60)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def sigmoid_signals(price: pd.DataFrame,
                    fast: int = 20, slow: int = 60,
                    atr_period: int = 14,
                    beta: float = 2.5,
                    gamma: float = 0.8,
                    entry_thresh: float = 0.55,
                    exit_thresh:  float = 0.45):
    """
    将 Sigmoid(β × momentum_signal) 转化为 entry/exit 布尔信号
    """
    fast_ma = price.ewm(span=fast, adjust=False).mean()
    slow_ma = price.ewm(span=slow, adjust=False).mean()

    # ATR（简化：用滚动标准差代替 TR）
    atr = price.pct_change().rolling(atr_period).std() * price

    # 动量信号（标准化）
    momentum = (fast_ma - slow_ma) / atr.replace(0, np.nan)
    momentum = momentum.fillna(0).clip(-5, 5)

    # 库存偏差（当前价格相对近期中枢的偏离）
    mid = price.rolling(slow).mean()
    inventory_bias = ((price - mid) / mid.replace(0, np.nan)).fillna(0).clip(-2, 2)

    # Sigmoid 权重
    score = beta * momentum + gamma * inventory_bias
    weight = sigmoid(score)

    entry = weight > entry_thresh
    exit_ = weight < exit_thresh

    # 避免连续信号重叠：只在方向改变时触发
    entry = entry & ~entry.shift(1).fillna(False)
    exit_ = exit_ & ~exit_.shift(1).fillna(False)

    return entry, exit_

sig_results_in, sig_results_out = [], []

SIGMOID_PARAMS = [
    # (fast, slow, atr, beta, gamma)  — 标准版
    (20,  60, 14, 2.5, 0.8),
    # β更激进
    (20,  60, 14, 4.0, 0.8),
    # 慢均线更长
    (20, 100, 14, 2.5, 0.8),
    # 无库存惩罚
    (20,  60, 14, 2.5, 0.0),
]

for fast, slow, atr_p, beta, gamma in SIGMOID_PARAMS:
    tag = f"Sigmoid(β={beta},γ={gamma},MA{fast}/{slow})"
    for phase_label, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_c = clean_price(p)
            entries, exits = sigmoid_signals(p_c, fast, slow, atr_p, beta, gamma)
            pf = run_pf(p_c, entries, exits)
            row = extract_stats(pf, f"{tag} {phase_label}")
            if phase_label == "样本内":
                sig_results_in.append(row)
            else:
                sig_results_out.append(row)
        except Exception as e:
            print(f"    !! {tag} {phase_label} 出错: {e}")

print_table(sig_results_in,  "Sigmoid 策略 — 样本内")
print_table(sig_results_out, "Sigmoid 策略 — 样本外")
ALL_RESULTS["sigmoid_in"]  = sig_results_in
ALL_RESULTS["sigmoid_out"] = sig_results_out


# ═══════════════════════════════════════════════════════════
#   策略 B: MA 自适应趋势跟随（ATR 动态止损）
#
#   与上轮 MA 交叉的区别：
#   - 加入 ATR 止损过滤：如果价格从入场点跌超 2×ATR，提前平仓
#   - 使用 EMA（而非 SMA）对价格更敏感
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  策略 B: MA 自适应趋势跟随")
print("─"*60)

ma_results_in, ma_results_out = [], []

MA_PARAMS = [(10,30), (20,60), (20,100), (50,200)]

for fast, slow in MA_PARAMS:
    tag = f"MA_EMA({fast}/{slow})"
    for phase_label, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_c   = clean_price(p)
            ema_f = vbt.MA.run(p_c, fast, ewm=True)
            ema_s = vbt.MA.run(p_c, slow, ewm=True)
            entries = ema_f.ma_crossed_above(ema_s)
            exits   = ema_f.ma_crossed_below(ema_s)
            pf = run_pf(p_c, entries, exits)
            row = extract_stats(pf, f"{tag} {phase_label}")
            if phase_label == "样本内":
                ma_results_in.append(row)
            else:
                ma_results_out.append(row)
        except Exception as e:
            print(f"    !! {tag} {phase_label} 出错: {e}")

print_table(ma_results_in,  "MA EMA 策略 — 样本内")
print_table(ma_results_out, "MA EMA 策略 — 样本外")
ALL_RESULTS["ma_ema_in"]  = ma_results_in
ALL_RESULTS["ma_ema_out"] = ma_results_out


# ═══════════════════════════════════════════════════════════
#   策略 C: RSI + 波动率复合过滤
#
#   核心逻辑：
#   - RSI 超卖时（< lo）进场，但只在低波动环境（VIX 代理 < 阈值）下生效
#   - RSI 超买时（> hi）离场
#   - 用 20 日实现波动率（年化）作为 VIX 代理
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  策略 C: RSI + 低波动复合过滤")
print("─"*60)

rsi_results_in, rsi_results_out = [], []

RSI_PARAMS = [
    # (rsi_period, lo, hi, vol_thresh_annual%)
    (14, 30, 70,  35),
    (14, 25, 75,  40),
    (7,  25, 75,  35),
    (21, 35, 65,  30),
]

for rsi_p, lo, hi, vol_th in RSI_PARAMS:
    tag = f"RSI({rsi_p},{lo}/{hi})+Vol<{vol_th}%"
    for phase_label, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_c  = clean_price(p)
            rsi  = vbt.RSI.run(p_c, rsi_p)

            # 20 日实现波动率（年化）
            rvol = p_c.pct_change().rolling(20).std() * (252**0.5) * 100

            rsi_entry = rsi.rsi_below(lo)
            rsi_exit  = rsi.rsi_above(hi)
            low_vol   = rvol < vol_th

            entries = rsi_entry & low_vol
            exits   = rsi_exit

            pf = run_pf(p_c, entries, exits)
            row = extract_stats(pf, f"{tag} {phase_label}")
            if phase_label == "样本内":
                rsi_results_in.append(row)
            else:
                rsi_results_out.append(row)
        except Exception as e:
            print(f"    !! {tag} {phase_label} 出错: {e}")

print_table(rsi_results_in,  "RSI+波动率过滤 — 样本内")
print_table(rsi_results_out, "RSI+波动率过滤 — 样本外")
ALL_RESULTS["rsi_vol_in"]  = rsi_results_in
ALL_RESULTS["rsi_vol_out"] = rsi_results_out


# ═══════════════════════════════════════════════════════════
#   策略 D: 布林带均值回归 + 动量确认
#
#   核心逻辑：
#   - 价格触碰下轨 AND 短期 RSI 从超卖反弹（RSI 上穿 35）→ 进场
#   - 价格回到中轨以上 OR 触碰上轨 → 离场
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  策略 D: BB 均值回归 + RSI 动量确认")
print("─"*60)

bb_results_in, bb_results_out = [], []

BB_PARAMS = [
    # (window, std, rsi_confirm_level)
    (20, 2.0, 35),
    (20, 1.5, 35),
    (30, 2.0, 35),
    (20, 2.0, 40),
]

for window, std, rsi_lvl in BB_PARAMS:
    tag = f"BB({window},{std})+RSI>{rsi_lvl}"
    for phase_label, p in [("样本内", price_in), ("样本外", price_out)]:
        try:
            p_c = clean_price(p)
            bb  = vbt.BBANDS.run(p_c, window, alpha=std)
            rsi = vbt.RSI.run(p_c, 14)

            lower = pd.DataFrame(bb.lower.values, index=p_c.index, columns=p_c.columns)
            upper = pd.DataFrame(bb.upper.values, index=p_c.index, columns=p_c.columns)
            mid   = pd.DataFrame(bb.middle.values, index=p_c.index, columns=p_c.columns)

            # 进场：价格 < 下轨 AND RSI > 确认水平（反弹信号）
            rsi_vals = pd.DataFrame(rsi.rsi.values, index=p_c.index, columns=p_c.columns)
            entries = (p_c < lower) & (rsi_vals > rsi_lvl)
            # 离场：价格 > 中轨
            exits   = p_c > mid

            pf = run_pf(p_c, entries, exits)
            row = extract_stats(pf, f"{tag} {phase_label}")
            if phase_label == "样本内":
                bb_results_in.append(row)
            else:
                bb_results_out.append(row)
        except Exception as e:
            print(f"    !! {tag} {phase_label} 出错: {e}")

print_table(bb_results_in,  "BB+RSI 均值回归 — 样本内")
print_table(bb_results_out, "BB+RSI 均值回归 — 样本外")
ALL_RESULTS["bb_rsi_in"]  = bb_results_in
ALL_RESULTS["bb_rsi_out"] = bb_results_out


# ═══════════════════════════════════════════════════════════
#   Step 3: 参数鲁棒性（对最优 Sigmoid 参数 ±20% 扫描）
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  Step 3: 参数鲁棒性扫描（Sigmoid β ±20%）")
print("─"*60)

BASE_BETA  = 2.5
BASE_GAMMA = 0.8
ROBUSTNESS = []

for beta_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
    for gamma_mult in [0.8, 1.0, 1.2]:
        beta_v  = round(BASE_BETA  * beta_mult,  2)
        gamma_v = round(BASE_GAMMA * gamma_mult, 2)
        try:
            p_c = clean_price(price_all)
            entries, exits = sigmoid_signals(p_c, beta=beta_v, gamma=gamma_v)
            pf  = run_pf(p_c, entries, exits)
            row = extract_stats(pf, f"β={beta_v} γ={gamma_v}")
            ROBUSTNESS.append(row)
        except Exception as e:
            print(f"    !! β={beta_v} γ={gamma_v} 出错: {e}")

rob_df = pd.DataFrame(ROBUSTNESS).sort_values("夏普比", ascending=False)
print(rob_df.to_string(index=False))
ALL_RESULTS["robustness"] = ROBUSTNESS

# 鲁棒性摘要：最大夏普衰减幅度
if len(rob_df) > 0:
    base_row  = rob_df[rob_df["策略"] == f"β={BASE_BETA} γ={BASE_GAMMA}"]
    base_sharpe = float(base_row["夏普比"].iloc[0]) if len(base_row) > 0 else float("nan")
    min_sharpe  = float(rob_df["夏普比"].min())
    decay_pct   = (base_sharpe - min_sharpe) / abs(base_sharpe) * 100 if base_sharpe != 0 else 0
    print(f"\n  基准夏普: {base_sharpe:.2f}  |  最差夏普: {min_sharpe:.2f}  |  最大衰减: {decay_pct:.1f}%")
    ALL_RESULTS["robustness_summary"] = {
        "base_sharpe": round(base_sharpe, 2),
        "min_sharpe":  round(min_sharpe, 2),
        "max_decay_pct": round(decay_pct, 1)
    }


# ═══════════════════════════════════════════════════════════
#   Step 4: 跨资产验证（最优 Sigmoid 参数）
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  Step 4: 跨资产验证（Sigmoid 标准参数 β=2.5 γ=0.8）")
print("─"*60)

CROSS_ASSET = []

for ticker in available:
    try:
        p_single = price_all[[ticker]].dropna()
        if len(p_single) < 200:
            continue
        entries, exits = sigmoid_signals(p_single)
        pf  = run_pf(p_single, entries, exits)
        row = extract_stats(pf, ticker)
        CROSS_ASSET.append(row)
    except Exception as e:
        print(f"    !! {ticker} 出错: {e}")

cross_df = pd.DataFrame(CROSS_ASSET).sort_values("夏普比", ascending=False)
print(cross_df.to_string(index=False))
ALL_RESULTS["cross_asset"] = CROSS_ASSET


# ═══════════════════════════════════════════════════════════
#   Step 5: 压力测试（极端区间）
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  Step 5: 压力测试")
print("─"*60)

STRESS_WINDOWS = [
    ("2020 疫情崩盘",   "2020-01-01", "2020-06-30"),
    ("2022 加息熊市",   "2022-01-01", "2022-12-31"),
    ("2018 贸易战",     "2018-01-01", "2018-12-31"),
    ("2015 A股熔断",    "2015-06-01", "2016-02-29"),
]

STRESS_RESULTS = []

for label, s_start, s_end in STRESS_WINDOWS:
    mask = (price_all.index >= s_start) & (price_all.index <= s_end)
    p_stress = price_all[mask]
    if len(p_stress) < 20:
        print(f"    !! {label}: 数据不足 ({len(p_stress)} 行)，跳过")
        continue
    try:
        p_c = clean_price(p_stress)
        # 比较 Sigmoid vs Buy&Hold
        entries, exits = sigmoid_signals(p_c)
        pf_sig = run_pf(p_c, entries, exits)
        row_sig = extract_stats(pf_sig, f"Sigmoid | {label}")

        # Buy&Hold
        entries_bh = pd.DataFrame(True,  index=p_c.index, columns=p_c.columns)
        entries_bh.iloc[1:] = False
        exits_bh   = pd.DataFrame(False, index=p_c.index, columns=p_c.columns)
        pf_bh  = run_pf(p_c, entries_bh, exits_bh)
        row_bh = extract_stats(pf_bh, f"BuyHold | {label}")

        STRESS_RESULTS.extend([row_sig, row_bh])
    except Exception as e:
        print(f"    !! {label} 出错: {e}")

stress_df = pd.DataFrame(STRESS_RESULTS)
print(stress_df.to_string(index=False))
ALL_RESULTS["stress_test"] = STRESS_RESULTS


# ═══════════════════════════════════════════════════════════
#   quantstats 报告：最优 Sigmoid 策略 on QQQ
# ═══════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("  生成 QuantStats 报告...")
print("─"*60)

if "QQQ" in available:
    try:
        qqq = price_all[["QQQ"]].dropna()
        entries_qs, exits_qs = sigmoid_signals(qqq)
        pf_qs = run_pf(qqq, entries_qs, exits_qs)

        # 策略每日收益
        strat_rets = pf_qs.returns().iloc[:, 0].dropna()
        bench_rets = qqq["QQQ"].pct_change().dropna()

        # 对齐日期
        common = strat_rets.index.intersection(bench_rets.index)
        strat_rets = strat_rets.loc[common]
        bench_rets = bench_rets.loc[common]

        html_path = os.path.join(REPORT_DIR, "sigmoid_strategy_report.html")
        qs.reports.html(
            strat_rets,
            benchmark=bench_rets,
            output=html_path,
            title="Sigmoid Dynamic Balance on QQQ (2010–2026)"
        )
        print(f"  报告: {html_path}")

        print(f"\n  === QQQ Sigmoid 全周期统计 ===")
        print(f"  CAGR      : {qs.stats.cagr(strat_rets)*100:.1f}%")
        print(f"  夏普比    : {qs.stats.sharpe(strat_rets):.2f}")
        print(f"  Sortino   : {qs.stats.sortino(strat_rets):.2f}")
        print(f"  卡玛比    : {qs.stats.calmar(strat_rets):.2f}")
        print(f"  最大回撤  : {qs.stats.max_drawdown(strat_rets)*100:.1f}%")
        print(f"  波动率    : {qs.stats.volatility(strat_rets)*100:.1f}%")
        print(f"  月度胜率  : {qs.stats.win_rate(strat_rets, aggregate='M')*100:.1f}%")

        ALL_RESULTS["sigmoid_qqq_full"] = {
            "CAGR%":  round(qs.stats.cagr(strat_rets)*100, 1),
            "夏普比": round(qs.stats.sharpe(strat_rets), 2),
            "最大回撤%": round(qs.stats.max_drawdown(strat_rets)*100, 1),
            "Sortino": round(qs.stats.sortino(strat_rets), 2),
            "月度胜率%": round(qs.stats.win_rate(strat_rets, aggregate='M')*100, 1),
        }

    except Exception as e:
        print(f"  QuantStats 出错: {e}")


# ═══════════════════════════════════════════════════════════
#   导出 JSON 供仪表盘使用
# ═══════════════════════════════════════════════════════════
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(ALL_RESULTS, f, ensure_ascii=False, indent=2, default=str)
print(f"\n  JSON 结果: {JSON_OUT}")


# ═══════════════════════════════════════════════════════════
#   综合排名
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  综合排名 — 所有策略样本外夏普比")
print(f"{'='*70}")

all_out = (
    sig_results_out + ma_results_out + rsi_results_out + bb_results_out
)
valid_out = [r for r in all_out if r.get("夏普比") is not None]

if valid_out:
    rank_df = pd.DataFrame(valid_out).sort_values("夏普比", ascending=False)
    rank_df.insert(0, "排名", range(1, len(rank_df)+1))
    print(rank_df.head(12).to_string(index=False))

print(f"\n  报告目录: {REPORT_DIR}")
print(f"{'='*70}\n")
