# Gate.io 交易所迁移 — Agent 任务提示词

## 目标

将现有的 OKX 实盘交易系统 (`live/` 目录) 迁移到 Gate.io 交易所，同时保持回测/训练管线不变。迁移分两个阶段：**阶段一：历史数据管线**（下载 Gate 历史数据 → 构建 dataset）和 **阶段二：实盘交易系统**（替换 OKX API 为 Gate API）。

---

## 阶段一：Gate 历史数据下载与管线构建

### 1.1 数据源说明

Gate 提供按月份的历史数据下载，URL 模式：
```
https://download.gatedata.org/${biz}/${type}/${year}${month}/${market}-${year}${month}.csv.gz
```

| 变量 | 说明 |
|------|------|
| `biz` | `futures_usdt`（USDT 永续合约） |
| `type` | `trades`（成交）、`orderbooks`（深度）、`candlesticks_1m`（1分钟K线） |
| `year+month` | 如 `202401` |
| `market` | 如 `ETH_USDT` |

**示例 URL：**
- 成交：`https://download.gatedata.org/futures_usdt/trades/202401/ETH_USDT-202401.csv.gz`
- 深度：`https://download.gatedata.org/futures_usdt/orderbooks/202401/ETH_USDT-202401.csv.gz`
- K线：`https://download.gatedata.org/futures_usdt/candlesticks_1m/202401/ETH_USDT-202401.csv.gz`

### 1.2 文件格式

**合约成交 (trades)：** CSV 列 = `timestamp, dealid, price, size`
- `size` 正数=多头(买), 负数=空头(卖)
- `timestamp` 为 Unix 秒（可能带小数）

**合约深度 (orderbooks)：** CSV 列 = `timestamp, action, price, size, begin-id, merged`
- `action`: `set`（全量快照）、`take`/`make`（增量更新）
- `size` 正数=买方, 负数=卖方
- 每小时一个 `set` 快照，之间是 100ms 粒度增量

**K线 (candlesticks_1m)：** CSV 列 = `timestamp, volume, close, high, low, open, amount`

### 1.3 需要创建的脚本：`download_gate_data.py`

在项目根目录创建下载脚本，功能需求：

1. **参数化**：支持 `--market`（默认 `ETH_USDT`）、`--start-month`（如 `202301`）、`--end-month`（如 `202603`）、`--types`（默认 `trades,orderbooks,candlesticks_1m`）、`--output-dir`（默认 `gate_data/`）
2. **下载逻辑**：遍历月份范围，构建 URL，用 `httpx` 或 `requests` 下载 `.csv.gz` 文件到 `gate_data/raw/` 子目录，跳过已存在的文件
3. **解压**：下载后自动 `gunzip` 解压为 `.csv`
4. **进度**：打印每个文件的下载进度和状态
5. **错误处理**：某月数据不存在时（404）跳过并记录

### 1.4 需要创建的脚本：`build_gate_dataset.py`

将下载的原始 Gate 数据转换为与现有 `dataset_enhanced.parquet` 相同格式的 parquet 文件，步骤：

1. **加载 K线数据** → 构建 1 分钟 OHLCV DataFrame，列名映射为 `datetime, open, high, low, close, volume`
2. **加载成交数据** → 按分钟聚合计算微观结构特征：
   - `aggressor_ratio`：买方成交量 / 总成交量（`size > 0` 为买方）
   - `net_taker_vol_ratio`：净买方量 / 总量
   - `trade_gini`：交易大小基尼系数
   - `large_trade_vol_ratio`：大单占比（size > 均值 * 2）
   - `trade_intensity`：`log1p(成交量)`
3. **加载深度数据** → 重建每分钟的 orderbook 快照，计算：
   - `obi`、`obi_1`、`obi_5`、`obi_20`：Order Book Imbalance
   - `ob_spread_bps`：买卖价差（基点）
   - `ob_depth_bid/ask_1/5/20`：各档位深度
   - `ob_microprice`：微观价格
   - 其他 `ob_*` 特征（参考 `live/orderbook_state.py` 的 `get_features()` 方法输出的完整字段列表）
4. **合并**：将 K线 + 成交特征 + 深度特征按时间戳 merge
5. **添加 scanner 标签**：复用 `quant_modeling.py` 中的 scanner 逻辑，调用 `pipline_modified.py` 的 `add_features()` 函数计算技术指标
6. **输出**：保存为 `dataset_gate_enhanced.parquet`，格式与 `dataset_enhanced.parquet` 完全一致

**关键约束**：
- 时间戳统一为 UTC datetime index
- 列名必须与现有 parquet 完全一致，以便 `train_xgb.py` 无需修改即可训练
- 数据中添加 `exchange` 列标记来源为 `"gate"`
- 保留 `scanner_variant` 和 `event_dir` 列

### 1.5 验证

创建完成后运行：
```bash
python download_gate_data.py --market ETH_USDT --start-month 202401 --end-month 202603
python build_gate_dataset.py --input-dir gate_data/ --output dataset_gate_enhanced.parquet
```

然后用现有训练脚本验证数据兼容性：
```bash
python train_xgb.py  # 应能无修改地在 Gate 数据上训练
```

---

## 阶段二：实盘交易系统迁移 (OKX → Gate)

### 2.1 架构说明

当前 `live/` 目录下的模块及其 OKX 依赖：

| 模块 | 职责 | OKX 依赖程度 |
|------|------|-------------|
| `config.py` | API 密钥、URL、交易参数 | **重度** — OKX URL、API Key、INST_ID 格式 |
| `ws_client.py` | WebSocket 连接（公共+私有） | **重度** — OKX WS 协议、登录签名（HMAC-SHA256）、消息格式 |
| `execution.py` | REST 下单/撤单/查持仓 | **重度** — OKX REST 签名、API 路径、请求/响应格式 |
| `orderbook_state.py` | 解析盘口推送维护状态 | **中度** — OKX `books5` 数据格式 (`bids`/`asks` 数组) |
| `bar_aggregator.py` | 逐笔成交聚合为K线 | **中度** — OKX trade 消息字段 (`px`, `sz`, `side`, `ts`) |
| `main.py` | 主循环、WS 订阅、消息路由 | **重度** — OKX channel 名称、消息解析、warmup candles |
| `feature_engine.py` | 特征计算 | **无** — 纯计算，不依赖交易所 |
| `scanner.py` | 信号扫描 | **无** — 纯计算 |
| `model_inference.py` | 模型推理 | **无** — 纯计算 |
| `event_aligner.py` | 事件对齐特征 | **无** — 纯计算 |
| `state.py` | 运行时状态 | **无** |

### 2.2 `config.py` 改造

**替换内容：**
```
OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE
→ GATE_API_KEY, GATE_API_SECRET（从环境变量读取：GATE_API_KEY, GATE_SECRET_KEY）

INST_ID = "ETH-USDT-SWAP" → INST_ID = "ETH_USDT"（Gate 合约交易对格式）
SETTLE = "usdt"  # 新增，Gate futures 按 settle 分路由

REST_BASE = "https://www.okx.com" → REST_BASE = "https://api.gateio.ws"
WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public" → WS_PUBLIC = "wss://fx-ws.gateio.ws/v4/ws/usdt"
WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private" → WS_PRIVATE = "wss://fx-ws.gateio.ws/v4/ws/usdt"（同一 URL，通过 auth 区分）

TD_MODE, FLAG → 删除（Gate 不需要这些概念）
```

### 2.3 `ws_client.py` 改造

**OKX 协议：**
- 订阅：`{"op": "subscribe", "args": [{"channel": "trades", "instId": "ETH-USDT-SWAP"}]}`
- 推送：`{"data": [...], "arg": {"channel": "..."}}`
- 私有频道：先发 login 消息（HMAC-SHA256 签名）

**Gate 协议（按文档改写）：**
- 订阅：`{"time": <unix_ts>, "channel": "futures.trades", "event": "subscribe", "payload": ["ETH_USDT"]}`
- 推送：`{"channel": "futures.trades", "event": "update", "result": [...]}`
- 私有频道：在请求 body 中带 `auth` 字段（HMAC-SHA512 签名），签名方式：`HexEncode(HMAC_SHA512(secret, "channel=<channel>&event=<event>&time=<time>"))`
- 心跳：Gate 使用 `futures.ping` / `futures.pong` 应用层心跳

**需要重写的类：**
- `OKXWebSocket` → `GateWebSocket`：
  - `_subscribe` 方法改为 Gate 格式
  - 消息解析从 `data["data"]` 改为 `data["result"]`
  - 添加应用层 ping/pong
- `OKXPrivateWebSocket` → `GatePrivateWebSocket`：
  - 登录签名从 HMAC-SHA256 改为 HMAC-SHA512
  - 签名字符串从 `ts + "GET" + "/users/self/verify"` 改为 `"channel=<ch>&event=<ev>&time=<ts>"`

### 2.4 `execution.py` 改造

**OKX REST 签名**：`HMAC-SHA256(secret, timestamp + method + path + body)`，Header 带 `OK-ACCESS-KEY`, `OK-ACCESS-SIGN`, `OK-ACCESS-TIMESTAMP`, `OK-ACCESS-PASSPHRASE`

**Gate REST 签名**：`HexEncode(HMAC_SHA512(secret, method + "\n" + url + "\n" + query_string + "\n" + HexEncode(SHA512(body)) + "\n" + timestamp))`，Header 带 `KEY`, `SIGN`, `Timestamp`

**API 路径映射：**

| 功能 | OKX 路径 | Gate 路径 |
|------|---------|----------|
| 查持仓 | `GET /api/v5/account/positions` | `GET /api/v4/futures/usdt/positions` |
| 查余额 | `GET /api/v5/account/balance` | `GET /api/v4/futures/usdt/accounts` |
| 下市价单 | `POST /api/v5/trade/order` | `POST /api/v4/futures/usdt/orders` |
| 查订单 | `GET /api/v5/trade/order` | `GET /api/v4/futures/usdt/orders/{order_id}` |
| 撤单 | `DELETE` 对应路径 | `DELETE /api/v4/futures/usdt/orders/{order_id}` |
| K线 | `GET /api/v5/market/candles` | `GET /api/v4/futures/usdt/candlesticks` |
| 合约信息 | `GET /api/v5/public/instruments` | `GET /api/v4/futures/usdt/contracts/{contract}` |
| 设置杠杆 | `POST /api/v5/account/set-leverage` | `POST /api/v4/futures/usdt/positions/{contract}/leverage`（或通过 dual_mode/position API） |

**关键差异：**
- Gate 合约大小用 `size`（正=做多, 负=做空），不需要 `side` 字段
- Gate 下单时 `size` 正数=开多, 负数=开空
- Gate TP/SL 通过 `POST /api/v4/futures/usdt/price_orders` 下条件单实现（不是 attachAlgoOrds）
- Gate 没有 OKX 的 `close-position` 接口，平仓需要下反向等量单
- Gate 合约面值 (`quanto_multiplier`) 需从合约信息获取

**`place_market_order` 改写要点：**
1. `side` 不存在了，用 `size` 的正负表示方向
2. 市价单：`price` 设为 `"0"`, `tif` 设为 `"ioc"`
3. TP/SL：下完主单后，单独调用 `POST /api/v4/futures/usdt/price_orders` 下止盈止损条件单

### 2.5 `orderbook_state.py` 改造

**OKX 格式**（`books5` 推送）：
```json
{"bids": [["3001.1", "1.5", "0", "2"], ...], "asks": [...]}
```

**Gate 格式**（`futures.order_book` 推送）：
```json
{"asks": [{"p": "3001.1", "s": 100}, ...], "bids": [{"p": "3000.9", "s": 200}, ...]}
```

改写 `update(data)` 方法：从 `data["asks"][i]["p"]` / `data["asks"][i]["s"]` 解析价格和数量。

### 2.6 `bar_aggregator.py` 改造

**OKX trade 字段**：`{"px": "3001.1", "sz": "0.5", "side": "buy", "ts": "1234567890000"}`

**Gate trade 字段**（WS `futures.trades` 推送）：`{"contract": "ETH_USDT", "size": 10, "price": "3001.1", "id": 123, "create_time": 1234567890, "create_time_ms": 1234567890123}`

改写 `ingest_trade`：
- `price` = `float(trade["price"])`
- `size` = `abs(trade["size"])`（Gate size 带符号）
- `side` = `"buy"` if `trade["size"] > 0` else `"sell"`
- `ts_ms` = `trade["create_time_ms"]`

### 2.7 `main.py` 改造

**WS 订阅频道映射：**
```python
# OKX
ws_trades: {"channel": "trades", "instId": "ETH-USDT-SWAP"}
ws_books:  {"channel": "books5", "instId": "ETH-USDT-SWAP"}
ws_private: [{"channel": "account"}, {"channel": "positions", "instType": "SWAP"}, {"channel": "balance_and_position"}]

# Gate（改为）
ws_trades: channel="futures.trades", payload=["ETH_USDT"]
ws_books:  channel="futures.order_book", payload=["ETH_USDT", "20", "0"]
ws_private: [channel="futures.orders", channel="futures.balances", channel="futures.positions"]
```

**消息处理函数改写：**

`on_trades_message(msg)`：
- OKX：`msg["data"][i]` 取 `px`, `sz`, `side`, `ts`
- Gate：`msg["result"][i]` 取 `price`, `size`(带符号), `create_time_ms`

`on_books_message(msg)`：
- OKX：`msg["data"][i]` 取 `bids`/`asks` 二维数组
- Gate：`msg["result"]` 取 `asks`/`bids` 对象数组

`on_private_message(msg)`：
- Gate 的 `futures.positions` 推送格式不同，需适配 `_process_position_data`
- Gate 的 `futures.balances` 推送需适配 `state.last_account` 更新

**warmup_from_candles()：**
- OKX：`GET /api/v5/market/candles` 返回 `[ts, o, h, l, c, vol, ...]`
- Gate：`GET /api/v4/futures/usdt/candlesticks?contract=ETH_USDT&interval=1m&limit=300`，返回 `[{"t":..., "o":..., "h":..., "l":..., "c":..., "v":...}]`

### 2.8 不需要修改的模块

以下模块是纯计算逻辑，**不要修改**：
- `feature_engine.py` — 调用 `pipline_modified.add_features()`
- `scanner.py` — 纯特征阈值判断
- `model_inference.py` — XGBoost/LGB/CatBoost 推理
- `event_aligner.py` — tick 数据特征计算
- `state.py` — dataclass 状态
- `templates/index.html` — 仪表盘前端（与交易所无关）

### 2.9 环境变量

迁移后需要的环境变量：
```bash
export GATE_API_KEY="your_gate_api_key"
export GATE_SECRET_KEY="your_gate_api_secret"
```

移除 OKX 相关环境变量（`OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE`）。Gate 不需要 passphrase。

---

## 约束与注意事项

1. **不要修改训练管线**：`train_xgb.py`, `train_multi_model.py`, `quant_modeling.py`, `pipline_modified.py`, `robust_oos_search.py` 等保持不变
2. **`best_config.json` 不变**：模型参数（h, tp, sl, threshold 等）与交易所无关
3. **保持 feature parity**：Gate 实盘计算的特征必须与训练数据特征完全一致
4. **合约面值差异**：OKX ETH-USDT-SWAP 面值 0.01 ETH/张；Gate ETH_USDT 面值需从 API `GET /api/v4/futures/usdt/contracts/ETH_USDT` 获取 `quanto_multiplier` 字段。`_calc_position_size()` 中的 `_ct_val` 需据此更新
5. **手续费**：Gate 合约默认 taker=0.05%, maker=0.02%。可用 GT 抵扣
6. **Gate API 签名用 SHA512**，不是 OKX 的 SHA256
7. **Gate 没有 `passphrase`**，只有 key + secret
8. **Gate 下单 size 正负表方向**，正数=做多，负数=做空。不需要 `side` 字段
9. **Gate 平仓**：没有 `close-position` 接口，需下反向等量单（`reduce_only: true`）
10. **Gate TP/SL**：用 `POST /api/v4/futures/usdt/price_orders` 下条件止盈止损单，不是 OKX 的 `attachAlgoOrds`
11. **运行环境**：`conda activate quant`，Python 3.11+
12. **安装依赖**：需要 `pip install gate-api`（Gate 官方 SDK，可选用；也可直接 httpx + 手动签名）
13. **测试**：用 Gate TestNet 先验证（WebSocket: `wss://fx-ws-testnet.gateio.ws/v4/ws/usdt`，REST: `https://fx-api-testnet.gateio.ws/api/v4`）
14. **Git**：每个阶段提交一次，commit message 格式如 `feat: add Gate historical data downloader`, `refactor: migrate live trading from OKX to Gate`

---

## 执行顺序

1. ✅ 创建 `download_gate_data.py`
2. ✅ 创建 `build_gate_dataset.py`
3. ✅ 下载数据并构建 parquet，验证与现有训练管线兼容
4. ✅ 改写 `live/config.py`
5. ✅ 改写 `live/ws_client.py`
6. ✅ 改写 `live/execution.py`
7. ✅ 改写 `live/orderbook_state.py`
8. ✅ 改写 `live/bar_aggregator.py`
9. ✅ 改写 `live/main.py`（消息路由、频道订阅、warmup）
10. ✅ 端到端测试：`python -m live.main` 启动，检查仪表盘、WebSocket 连接、信号触发
