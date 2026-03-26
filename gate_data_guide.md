# Gate.io 合约历史数据下载指南

## 一、概述

Gate.io 提供 `futures_usdt`（USDT 结算永续合约）的历史逐笔成交和深度数据，按月/小时打包为 `.csv.gz` 文件，可直接通过 URL 下载。

---

## 二、URL 规则

### 逐笔成交（trades） — 按月打包

```
https://download.gatedata.org/futures_usdt/trades/{YYYYMM}/{PAIR}-{YYYYMM}.csv.gz
```

### 深度快照（orderbooks） — 按小时打包

```
https://download.gatedata.org/futures_usdt/orderbooks/{YYYYMM}/{PAIR}-{YYYYMMDDHH}.csv.gz
```

### 变量说明

| 变量 | 格式 | 示例 |
|------|------|------|
| `PAIR` | 大写、下划线分隔 | `ETH_USDT`、`BTC_USDT` |
| `YYYYMM` | 年月 | `202602` |
| `YYYYMMDDHH` | 年月日时（UTC） | `2026030108` = 2026-03-01 08:00 UTC |

### URL 示例

```bash
# ETH 2026年2月 trades（整月，~340MB 压缩）
https://download.gatedata.org/futures_usdt/trades/202602/ETH_USDT-202602.csv.gz

# ETH 2026年3月1日 08:00 UTC 的 orderbook（~7-40MB 压缩）
https://download.gatedata.org/futures_usdt/orderbooks/202603/ETH_USDT-2026030108.csv.gz
```

---

## 三、数据可用性

| 数据类型 | 延迟 | 说明 |
|----------|------|------|
| **trades** | 月末出文件 | 当月数据需等到次月初才可下载（如3月数据约4月1日可用） |
| **orderbooks** | ~1-2小时 | 当天已过去的整点小时可下载（如 UTC 16:40 时，15:00 的文件已可用，16:00 的还不行） |

---

## 四、文件格式

所有文件无 header 行，直接是 CSV 数据。

### 4.1 Trades（逐笔成交）

**列顺序：** `timestamp, dealid, price, size`

```csv
1769904000.424738,687159738,2450.29,5
1769904000.829937,687159739,2450.14,-1283
1769904001.159264,687159741,2450.15,10
```

| 字段 | 说明 |
|------|------|
| `timestamp` | Unix 时间戳，秒.微秒精度 |
| `dealid` | 成交ID，递增 |
| `price` | 成交价格（USDT） |
| `size` | **正数 = taker buy（买方主动成交）**，**负数 = taker sell（卖方主动成交）**。绝对值为合约张数 |

### 4.2 Orderbooks（深度）

**列顺序：** `timestamp, action, price, size, begin-id, merged`

```csv
1774224000,set,2050.53,29398.0,92440547527,0
1774224000.1,make,2049.6,433.0,92440547528,1
1774224000.1,take,2052.93,-1868.0,92440547529,1
```

| 字段 | 说明 |
|------|------|
| `timestamp` | Unix 时间戳，`set` 行精确到秒，更新行精确到 100ms（如 `.1` = 100ms） |
| `action` | `set` = 每小时初始全量快照；`make` = 新挂单/改单；`take` = 被吃单成交 |
| `price` | 价格档位（USDT） |
| `size` | 见下方说明 |
| `begin-id` | 本批次起始事件ID |
| `merged` | 本行聚合的事件数量（0 或 1+） |

#### size 字段的含义

| action | size > 0 | size < 0 |
|--------|----------|----------|
| `set`（全量快照） | 全部为正数，**需根据价格判断 bid/ask**（低于中间价 = bid，高于中间价 = ask） |
| `make`（挂单更新） | **bid 侧**（买方挂单） | **ask 侧**（卖方挂单） |
| `take`（成交更新） | **bid 侧**（买方被吃） | **ask 侧**（卖方被吃） |

---

## 五、下载命令

### 单文件下载

```bash
# 下载 trades
curl -o ETH_USDT-202602.csv.gz \
  "https://download.gatedata.org/futures_usdt/trades/202602/ETH_USDT-202602.csv.gz"
gunzip ETH_USDT-202602.csv.gz

# 下载某小时 orderbook
curl -o ETH_USDT-2026030108.csv.gz \
  "https://download.gatedata.org/futures_usdt/orderbooks/202603/ETH_USDT-2026030108.csv.gz"
gunzip ETH_USDT-2026030108.csv.gz
```

### 批量下载某天全部 orderbook（24小时）

```bash
PAIR="ETH_USDT"
DATE="20260301"   # YYYYMMDD

for h in $(seq -w 0 23); do
  URL="https://download.gatedata.org/futures_usdt/orderbooks/${DATE:0:6}/${PAIR}-${DATE}${h}.csv.gz"
  echo "Downloading $URL ..."
  curl -fO "$URL" || echo "  [skip] $URL not available"
done
```

### 批量下载某月全部 orderbook

```bash
PAIR="ETH_USDT"
YEAR=2026
MONTH=02

for DAY in $(seq -w 1 28); do
  for HOUR in $(seq -w 0 23); do
    YM=$(printf "%04d%02d" $YEAR $MONTH)
    FULL="${YM}${DAY}${HOUR}"
    URL="https://download.gatedata.org/futures_usdt/orderbooks/${YM}/${PAIR}-${FULL}.csv.gz"
    curl -sf -O "$URL" && echo "OK: $FULL" || true
  done
done
```

### 流式查看（不落盘）

```bash
# 快速查看前 10 行 trades
curl -s "https://download.gatedata.org/futures_usdt/trades/202602/ETH_USDT-202602.csv.gz" \
  | gunzip | head -10

# 快速查看 orderbook 的 make/take 更新
curl -s "https://download.gatedata.org/futures_usdt/orderbooks/202603/ETH_USDT-2026030100.csv.gz" \
  | gunzip | awk -F',' '$2 != "set"' | head -20
```

---

## 六、注意事项

1. **文件体积**
   - Trades：ETH 一个月约 **340MB** 压缩 / **2-3GB** 解压
   - Orderbooks：每小时 **7-40MB** 压缩，一天约 **0.5-1GB** 压缩

2. **时区**
   - 所有时间戳均为 **UTC**
   - 小时编号也是 UTC：`2026030108` = 北京时间 2026-03-01 **16:00**

3. **合约面值**
   - Gate `ETH_USDT` 永续合约面值 = **1 USD/张**
   - OKX `ETH-USDT-SWAP` 面值 = **0.01 ETH/张**
   - 换算仓位时需注意：Gate 的 `size=1000` 对应 $1000 名义价值

4. **当月 trades 不可用**
   - 如需当月逐笔数据，可从 orderbook 的 `take` 行近似提取成交记录

5. **set 快照的 bid/ask 区分**
   - 全量快照中 size 全为正数，需找到市场中间价后按价格高低划分
   - 最高 bid ≈ 最低 ask，可从紧随其后的 `make`/`take` 更新推断中间价
