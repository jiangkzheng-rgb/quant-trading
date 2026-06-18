# 量化交易策略工作区

## 核心工具栈（已安装）

| 工具 | 版本 | 用途 |
|------|------|------|
| `yfinance` | 1.4.1 | 行情/历史/分红/财务数据 |
| `openbb` | 4.7.2 | 综合金融数据平台（股票/ETF/宏观/期权） |
| `edgartools` | 5.36 | SEC EDGAR 10-K/10-Q/8-K/Form4/13F 解析 |
| `sec-edgar-downloader` | 5.1 | 批量下载 SEC 文件 |
| `vectorbt` | 1.0.0 | 向量化策略回测（多参数/多资产/多周期） |
| `quantstats` | 0.0.81 | 绩效报告（夏普/回撤/胜率/月度收益） |
| `finrl` | 0.3.7 | 强化学习交易（需 gymnasium+sb3） |

## 工作原则

### 回测优先
- **每个策略必须先回测，再讨论实盘**
- 回测使用 `vectorbt`，至少覆盖 3 年以上历史
- 必须计算并展示：夏普比、最大回撤、年化收益、胜率
- 用 `quantstats.reports.html()` 生成完整 HTML 报告

### 多轮验证
执行以下顺序验证策略有效性：
1. **样本内回测**：用 2018-2022 数据拟合参数
2. **样本外验证**：用 2023-至今数据验证，参数不变
3. **参数鲁棒性**：在参数 ±20% 范围内批量回测，检验结果是否崩溃
4. **跨资产验证**：在至少 5 只股票/ETF 上测试同一策略
5. **压力测试**：包含 2020-03（疫情崩盘）、2022-01（加息熊市）等极端区间

### 数据获取优先级
```
yfinance（免费快速）→ openbb（更全面）→ edgartools（SEC 原文）
```

## 常用代码模板

### 快速拉取数据
```python
import yfinance as yf
import pandas as pd

def get_data(tickers, start="2018-01-01", end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return data["Close"]
```

### vectorbt 策略回测框架
```python
import vectorbt as vbt
import numpy as np

def backtest_ma_cross(ticker="QQQ", fast=20, slow=50, start="2018-01-01"):
    price = vbt.YFData.download(ticker, start=start).get("Close")
    fast_ma = vbt.MA.run(price, fast)
    slow_ma = vbt.MA.run(price, slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits  = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(price, entries, exits, freq="D")
    return pf

pf = backtest_ma_cross()
print(pf.stats())
```

### quantstats 报告生成
```python
import quantstats as qs
import yfinance as yf

returns = yf.download("QQQ", start="2018-01-01")["Close"].pct_change().dropna()
qs.reports.html(returns, benchmark="SPY", output="report.html", title="QQQ Strategy")
```

### OpenBB 数据拉取
```python
from openbb import obb

# 股价历史
hist = obb.equity.price.historical("AAPL", start_date="2020-01-01", provider="yfinance")
df = hist.to_df()

# 基本面
info = obb.equity.fundamental.overview("AAPL", provider="fmp")
```

### edgartools SEC 文件
```python
from edgar import Company, set_identity

set_identity("蒋坤正 438454658@qq.com")  # SEC 要求填写

company = Company("AAPL")
filings = company.get_filings(form="10-K")
latest_10k = filings.latest(1)
```

### 批量 SEC 下载
```python
from sec_edgar_downloader import Downloader

dl = Downloader("蒋坤正", "438454658@qq.com", "C:/Users/jiang/Downloads/sec_filings")
dl.get("10-K", "AAPL", limit=5)  # 下载最近 5 份年报
```

## 回测报告标准输出

每次回测必须输出以下指标：
- 年化收益率 (CAGR)
- 最大回撤 (Max Drawdown)  
- 夏普比率 (Sharpe Ratio)
- 卡玛比率 (Calmar Ratio)
- 胜率 (Win Rate)
- 盈亏比 (Profit Factor)
- 总交易次数

## 文件结构约定

```
outputs/
  strategies/        # 策略代码
    ma_cross.py
    rsi_reversal.py
  backtests/         # 回测结果 HTML
  data/              # 缓存的历史数据
  reports/           # quantstats 报告
```

## 连接盈立证券

账户信息：盈立号 8939758373，邮箱 438454658@qq.com
API 接入申请中（评估表已提交）
实盘对接待 API 审批通过后配置
