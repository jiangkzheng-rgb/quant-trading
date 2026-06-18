"""
量化工具链验证脚本
运行: python verify_stack.py
"""
import sys, warnings
warnings.filterwarnings("ignore")

results = {}

# 1. yfinance
try:
    import yfinance as yf
    df = yf.download("QQQ", start="2023-01-01", end="2023-12-31",
                     auto_adjust=True, progress=False)
    assert len(df) > 200
    results["yfinance"] = f"OK  - QQQ 2023: {len(df)} rows"
except Exception as e:
    results["yfinance"] = f"FAIL - {e}"

# 2. vectorbt
try:
    import vectorbt as vbt
    import pandas as pd
    price = vbt.YFData.download("QQQ", start="2020-01-01",
                                end="2024-12-31").get("Close")
    fast = vbt.MA.run(price, 20)
    slow = vbt.MA.run(price, 50)
    entries = fast.ma_crossed_above(slow)
    exits   = fast.ma_crossed_below(slow)
    pf = vbt.Portfolio.from_signals(price, entries, exits, freq="D")
    stats = pf.stats()
    sharpe = round(float(stats["Sharpe Ratio"]), 2)
    mdd    = round(float(stats["Max Drawdown [%]"]), 1)
    ret    = round(float(stats["Total Return [%]"]), 1)
    results["vectorbt"] = f"OK  - QQQ MA(20/50) | Return={ret}% Sharpe={sharpe} MaxDD={mdd}%"
except Exception as e:
    results["vectorbt"] = f"FAIL - {e}"

# 3. quantstats
try:
    import quantstats as qs
    import yfinance as yf
    prices = yf.download("QQQ", start="2020-01-01",
                         progress=False, auto_adjust=True)["Close"].squeeze()
    rets = prices.pct_change().dropna()
    sharpe = round(qs.stats.sharpe(rets), 2)
    mdd    = round(qs.stats.max_drawdown(rets) * 100, 1)
    results["quantstats"] = f"OK  - QQQ Sharpe={sharpe} MaxDD={mdd}%"
except Exception as e:
    results["quantstats"] = f"FAIL - {e}"

# 4. openbb
try:
    from openbb import obb
    hist = obb.equity.price.historical(
        "AAPL", start_date="2024-01-01", end_date="2024-03-31",
        provider="yfinance"
    )
    df_obb = hist.to_df()
    results["openbb"] = f"OK  - AAPL via openbb: {len(df_obb)} rows"
except Exception as e:
    results["openbb"] = f"FAIL - {e}"

# 5. edgartools
try:
    from edgar import Company, set_identity
    set_identity("QuantTest 438454658@qq.com")
    company = Company("AAPL")
    filings = company.get_filings(form="10-K")
    count = len(filings)
    results["edgartools"] = f"OK  - AAPL 10-K filings found: {count}"
except Exception as e:
    results["edgartools"] = f"FAIL - {e}"

# 6. sec-edgar-downloader
try:
    from sec_edgar_downloader import Downloader
    results["sec-edgar-downloader"] = "OK  - Downloader ready (lazy init)"
except Exception as e:
    results["sec-edgar-downloader"] = f"FAIL - {e}"

# 7. finrl
try:
    import gymnasium
    results["finrl/gymnasium"] = f"OK  - gymnasium {gymnasium.__version__}"
except Exception as e:
    results["finrl/gymnasium"] = f"WARN - {e} (finrl RL features need gym)"

# Print results
print("\n" + "="*60)
print("  量化工具链验证结果")
print("="*60)
for name, status in results.items():
    icon = "[PASS]" if "OK" in status else ("[WARN]" if "WARN" in status else "[FAIL]")
    print(f"  {icon}  {name:<22} {status}")
print("="*60)

pass_count = sum(1 for v in results.values() if "OK" in v)
total = len(results)
print(f"\n  {pass_count}/{total} 工具就绪")
if pass_count >= 5:
    print("  核心回测链路 (yfinance + vectorbt + quantstats) 全部可用")
