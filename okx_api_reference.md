# OKX API v5 参考手册

## 1. 概览

OKX API v5 统一了所有产品线（现货、合约、期权）的接口，同一个 endpoint 通过 `instType` 参数区分产品类型。

- 官方文档: https://www.okx.com/docs-v5/en/
- Python SDK (官方): `pip install python-okx` (https://github.com/okxapi/python-okx)
- 第三方 SDK: `pip install okx-sdk` (https://github.com/burakoner/okx-sdk)

## 2. 基础配置

### Base URLs

| 环境 | REST API | WebSocket Public | WebSocket Private | WebSocket Business |
|------|----------|-----------------|-------------------|-------------------|
| 实盘 | `https://www.okx.com` | `wss://ws.okx.com:8443/ws/v5/public` | `wss://ws.okx.com:8443/ws/v5/private` | `wss://ws.okx.com:8443/ws/v5/business` |
| 模拟盘 | `https://www.okx.com` (flag=1) | `wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999` | `wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999` | `wss://wspap.okx.com:8443/ws/v5/business?brokerId=9999` |

### 认证

需要 3 个凭证：`API Key`、`Secret Key`、`Passphrase`

REST 请求头：
```
OK-ACCESS-KEY: <api_key>
OK-ACCESS-SIGN: <signature>
OK-ACCESS-TIMESTAMP: <timestamp>
OK-ACCESS-PASSPHRASE: <passphrase>
x-simulated-trading: 1  # 模拟盘设为 "1"，实盘设为 "0"
Content-Type: application/json
```

### 签名生成 (HMAC-SHA256)

```python
import hmac, base64, datetime

# 1. 生成 ISO 时间戳
timestamp = datetime.datetime.utcnow().isoformat("T", "milliseconds") + "Z"

# 2. 拼接 prehash 字符串
# GET:  timestamp + "GET" + requestPath(含query string)
# POST: timestamp + "POST" + requestPath + body(JSON string)
prehash = timestamp + method.upper() + request_path + body

# 3. HMAC-SHA256 签名
mac = hmac.new(
    bytes(secret_key, encoding='utf8'),
    bytes(prehash, encoding='utf-8'),
    digestmod='sha256'
)
signature = base64.b64encode(mac.digest())
```

## 3. Python SDK 快速上手

```python
pip install python-okx
```

```python
from okx import Trade, Account, MarketData, PublicData

# 初始化 (flag="1" 模拟盘, "0" 实盘)
trade_api = Trade.TradeAPI(api_key, secret_key, passphrase, flag="1")
account_api = Account.AccountAPI(api_key, secret_key, passphrase, flag="1")
market_api = MarketData.MarketAPI(flag="1")        # 公开接口无需认证
public_api = PublicData.PublicAPI(flag="1")
```

## 4. REST API 完整端点列表

### 4.1 Trade (交易) — `/api/v5/trade/`

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v5/trade/order` | 下单 |
| POST | `/api/v5/trade/batch-orders` | 批量下单 (最多20笔) |
| POST | `/api/v5/trade/cancel-order` | 撤单 |
| POST | `/api/v5/trade/cancel-batch-orders` | 批量撤单 |
| POST | `/api/v5/trade/amend-order` | 改单 |
| POST | `/api/v5/trade/amend-batch-orders` | 批量改单 |
| POST | `/api/v5/trade/close-position` | 市价平仓 |
| GET | `/api/v5/trade/order` | 查询订单详情 |
| GET | `/api/v5/trade/orders-pending` | 查询未成交订单列表 |
| GET | `/api/v5/trade/orders-history` | 查询历史订单 (7天) |
| GET | `/api/v5/trade/orders-history-archive` | 查询历史订单 (3个月) |
| GET | `/api/v5/trade/fills` | 查询成交明细 |
| GET | `/api/v5/trade/fills-history` | 查询历史成交明细 |
| POST | `/api/v5/trade/order-algo` | 下策略委托单 (止盈止损/追踪/冰山/时间加权) |
| POST | `/api/v5/trade/cancel-algos` | 撤销策略委托单 |
| POST | `/api/v5/trade/amend-algos` | 修改策略委托单 |
| GET | `/api/v5/trade/order-algo` | 查询策略委托单详情 |
| GET | `/api/v5/trade/orders-algo-pending` | 查询未完成策略委托单 |
| GET | `/api/v5/trade/orders-algo-history` | 查询历史策略委托单 |
| GET | `/api/v5/trade/easy-convert-currency-list` | 闪兑币种列表 |
| POST | `/api/v5/trade/easy-convert` | 闪兑交易 |
| GET | `/api/v5/trade/easy-convert-history` | 闪兑历史 |
| GET | `/api/v5/trade/one-click-repay-currency-list` | 一键还债币种列表 |
| POST | `/api/v5/trade/one-click-repay` | 一键还债 |
| GET | `/api/v5/trade/one-click-repay-history` | 一键还债历史 |

#### 下单关键参数

```python
trade_api.place_order(
    instId="ETH-USDT",       # 产品ID
    tdMode="cash",            # 交易模式: cash(现货), cross(全仓), isolated(逐仓)
    side="buy",               # buy / sell
    ordType="limit",          # market/limit/post_only/fok/ioc/optimal_limit_ioc
    sz="0.1",                 # 数量
    px="2500",                # 价格 (limit单必填)
    posSide="",               # 持仓方向: long/short (双向持仓模式)
    tgtCcy="",                # 计价方式: base_ccy / quote_ccy
    clOrdId="",               # 自定义订单ID
    reduceOnly="false",       # 是否只减仓
    # 附带止盈止损
    attachAlgoOrds=[{
        "tpTriggerPx": "2600",
        "tpOrdPx": "-1",      # -1 表示市价
        "slTriggerPx": "2400",
        "slOrdPx": "-1",
    }]
)
```

#### 订单类型 (ordType)

| 类型 | 说明 |
|------|------|
| `market` | 市价单 |
| `limit` | 限价单 |
| `post_only` | 只做 Maker |
| `fok` | 全部成交或取消 |
| `ioc` | 立即成交并取消剩余 |
| `optimal_limit_ioc` | 最优限价 IOC |

#### 策略委托单类型 (algo ordType)

| 类型 | 说明 |
|------|------|
| `conditional` | 条件单 (止盈止损) |
| `oco` | OCO 单 |
| `trigger` | 计划委托 |
| `move_order_stop` | 移动止盈止损 |
| `twap` | 时间加权 |
| `chase` | 追单 |

### 4.2 Account (账户) — `/api/v5/account/`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v5/account/balance` | 查询账户余额 |
| GET | `/api/v5/account/positions` | 查询持仓 |
| GET | `/api/v5/account/positions-history` | 查询历史持仓 |
| GET | `/api/v5/account/account-position-risk` | 查询账户持仓风险 |
| GET | `/api/v5/account/bills` | 查询账单流水 (7天) |
| GET | `/api/v5/account/bills-archive` | 查询账单流水 (3个月) |
| GET | `/api/v5/account/config` | 查询账户配置 |
| POST | `/api/v5/account/set-position-mode` | 设置持仓模式 (单向/双向) |
| POST | `/api/v5/account/set-leverage` | 设置杠杆倍数 |
| GET | `/api/v5/account/leverage-info` | 查询杠杆倍数 |
| GET | `/api/v5/account/max-size` | 查询最大可交易数量 |
| GET | `/api/v5/account/max-avail-size` | 查询最大可用数量 |
| POST | `/api/v5/account/position/margin-balance` | 调整保证金 |
| GET | `/api/v5/account/max-loan` | 查询最大可借 |
| GET | `/api/v5/account/trade-fee` | 查询手续费费率 |
| GET | `/api/v5/account/interest-accrued` | 查询计息记录 |
| GET | `/api/v5/account/interest-rate` | 查询用户当前利率 |
| POST | `/api/v5/account/set-greeks` | 设置 Greeks 展示方式 |
| POST | `/api/v5/account/set-isolated-mode` | 设置逐仓模式 |
| GET | `/api/v5/account/max-withdrawal` | 查询最大可转出 |
| GET | `/api/v5/account/risk-state` | 查询账户风险状态 |
| POST | `/api/v5/account/borrow-repay` | 借币还币 |
| GET | `/api/v5/account/borrow-repay-history` | 借币还币历史 |
| GET | `/api/v5/account/interest-limits` | 查询借币利率与限额 |
| GET | `/api/v5/account/greeks` | 查询 Greeks |
| POST | `/api/v5/account/set-account-level` | 设置账户模式 |
| GET | `/api/v5/account/position-tiers` | 查询仓位档位 |

### 4.3 Market Data (行情) — `/api/v5/market/`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v5/market/tickers` | 获取所有产品行情 |
| GET | `/api/v5/market/ticker` | 获取单个产品行情 |
| GET | `/api/v5/market/index-tickers` | 获取指数行情 |
| GET | `/api/v5/market/books` | 获取深度 (orderbook) |
| GET | `/api/v5/market/books-lite` | 获取轻量深度 |
| GET | `/api/v5/market/candles` | 获取 K 线 |
| GET | `/api/v5/market/history-candles` | 获取历史 K 线 |
| GET | `/api/v5/market/index-candles` | 获取指数 K 线 |
| GET | `/api/v5/market/mark-price-candles` | 获取标记价格 K 线 |
| GET | `/api/v5/market/trades` | 获取最近成交 |
| GET | `/api/v5/market/history-trades` | 获取历史成交 |
| GET | `/api/v5/market/platform-24-volume` | 获取平台24小时成交量 |
| GET | `/api/v5/market/index-components` | 获取指数成分 |
| GET | `/api/v5/market/exchange-rate` | 获取法币汇率 |
| GET | `/api/v5/market/block-tickers` | 获取大宗交易行情 |
| GET | `/api/v5/market/block-ticker` | 获取单个大宗交易行情 |
| GET | `/api/v5/market/block-trades` | 获取大宗交易成交 |

### 4.4 Public Data (公共数据) — `/api/v5/public/`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v5/public/instruments` | 获取交易产品信息 (instType: SPOT/MARGIN/SWAP/FUTURES/OPTION) |
| GET | `/api/v5/public/delivery-exercise-history` | 获取交割/行权历史 |
| GET | `/api/v5/public/open-interest` | 获取持仓总量 |
| GET | `/api/v5/public/funding-rate` | 获取永续合约当前资金费率 |
| GET | `/api/v5/public/funding-rate-history` | 获取永续合约历史资金费率 |
| GET | `/api/v5/public/price-limit` | 获取限价 |
| GET | `/api/v5/public/opt-summary` | 获取期权定价 |
| GET | `/api/v5/public/estimated-price` | 获取预估交割/行权价格 |
| GET | `/api/v5/public/discount-rate-interest-free-quota` | 获取折扣率与免息额度 |
| GET | `/api/v5/public/time` | 获取服务器时间 |
| GET | `/api/v5/public/liquidation-orders` | 获取强平单 |
| GET | `/api/v5/public/mark-price` | 获取标记价格 |
| GET | `/api/v5/public/position-tiers` | 获取仓位档位 |
| GET | `/api/v5/public/interest-rate-loan-quota` | 获取杠杆利率和借币限额 |
| GET | `/api/v5/public/underlying` | 获取标的指数 |
| GET | `/api/v5/public/insurance-fund` | 获取风险准备金余额 |
| GET | `/api/v5/public/convert-contract-coin` | 张币转换 |

### 4.5 Funding (资金) — `/api/v5/asset/`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v5/asset/balances` | 查询资金账户余额 |
| GET | `/api/v5/asset/deposit-address` | 获取充值地址 |
| GET | `/api/v5/asset/deposit-history` | 查询充值记录 |
| POST | `/api/v5/asset/withdrawal` | 提币 |
| GET | `/api/v5/asset/withdrawal-history` | 查询提币记录 |
| POST | `/api/v5/asset/cancel-withdrawal` | 撤销提币 |
| POST | `/api/v5/asset/transfer` | 资金划转 (资金账户↔交易账户) |
| GET | `/api/v5/asset/transfer-state` | 查询划转状态 |
| GET | `/api/v5/asset/currencies` | 获取币种列表 |
| GET | `/api/v5/asset/bills` | 查询资金流水 |
| GET | `/api/v5/asset/asset-valuation` | 查询资产估值 |
| GET | `/api/v5/asset/deposit-withdraw-status` | 查询充提状态 |
| POST | `/api/v5/asset/convert-dust-assets` | 小额资产兑换 |
| GET | `/api/v5/asset/non-tradable-assets` | 查询不可交易资产 |
| GET | `/api/v5/asset/convert/currencies` | 闪兑币种列表 |
| GET | `/api/v5/asset/convert/currency-pair` | 闪兑币对信息 |
| GET | `/api/v5/asset/convert/estimate-quote` | 闪兑预估报价 |
| POST | `/api/v5/asset/convert/trade` | 闪兑交易 |
| GET | `/api/v5/asset/convert/history` | 闪兑历史 |

### 4.6 Trading Data (交易大数据) — `/api/v5/rubik/stat/`

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v5/rubik/stat/trading-data/support-coin` | 获取支持的币种 |
| GET | `/api/v5/rubik/stat/taker-volume` | 获取主动买入/卖出量 |
| GET | `/api/v5/rubik/stat/margin/loan-ratio` | 获取杠杆多空比 |
| GET | `/api/v5/rubik/stat/contracts/long-short-account-ratio` | 获取合约多空持仓人数比 |
| GET | `/api/v5/rubik/stat/contracts/open-interest-volume` | 获取合约持仓量及交易量 |
| GET | `/api/v5/rubik/stat/contracts/open-interest-history` | 获取合约持仓量历史 |
| GET | `/api/v5/rubik/stat/option/open-interest-volume` | 获取期权持仓量及交易量 |
| GET | `/api/v5/rubik/stat/option/open-interest-volume-ratio` | 获取期权看涨/看跌比 |
| GET | `/api/v5/rubik/stat/option/open-interest-volume-expiry` | 按到期日分持仓量 |
| GET | `/api/v5/rubik/stat/option/open-interest-volume-strike` | 按行权价分持仓量 |
| GET | `/api/v5/rubik/stat/option/taker-block-volume` | 获取期权 Taker 大单量 |

### 4.7 其他模块

| 模块 | 前缀 | 说明 |
|------|------|------|
| SubAccount | `/api/v5/users/subaccount/`, `/api/v5/asset/subaccount/` | 子账户管理 |
| Grid Trading | `/api/v5/tradingBot/grid/` | 网格交易机器人 |
| Recurring Buy | `/api/v5/tradingBot/recurring/` | 定投策略 |
| Copy Trading | `/api/v5/copytrading/` | 跟单交易 |
| Block Trading | `/api/v5/rfq/` | 大宗交易 (RFQ) |
| Spread Trading | `/api/v5/sprd/` | 价差交易 |
| Finance/Staking | `/api/v5/finance/staking-defi/`, `/api/v5/finance/savings/` | 质押/理财 |
| Flexible Loan | `/api/v5/finance/flexible-loan/` | 灵活借贷 |
| System Status | `/api/v5/system/status` | 系统状态 |

## 5. WebSocket API

### 5.1 三个通道

| 通道 | URL (实盘) | 认证 | 用途 |
|------|-----------|------|------|
| Public | `wss://ws.okx.com:8443/ws/v5/public` | 不需要 | 行情、深度、成交、K线 |
| Private | `wss://ws.okx.com:8443/ws/v5/private` | 需要 | 账户、持仓、订单推送 |
| Business | `wss://ws.okx.com:8443/ws/v5/business` | 需要 | 策略委托、高级K线、标记价格K线 |

### 5.2 订阅格式

```json
{
  "op": "subscribe",
  "args": [
    {"channel": "tickers", "instId": "ETH-USDT"},
    {"channel": "books5", "instId": "ETH-USDT"}
  ]
}
```

### 5.3 常用 Public 频道

| 频道 | 说明 |
|------|------|
| `tickers` | 行情推送 |
| `books` | 深度推送 (全量+增量) |
| `books5` | 5档深度 |
| `books-l2-tbt` | 逐笔深度 (400档) |
| `books50-l2-tbt` | 逐笔深度 (50档) |
| `trades` | 成交推送 |
| `candle{period}` | K线 (如 candle1m, candle5m, candle1H) |
| `mark-price` | 标记价格 |
| `funding-rate` | 资金费率 |
| `open-interest` | 持仓总量 |
| `opt-summary` | 期权定价 |

### 5.4 常用 Private 频道

| 频道 | 说明 |
|------|------|
| `account` | 账户信息推送 |
| `positions` | 持仓推送 |
| `orders` | 订单推送 |
| `orders-algo` | 策略委托单推送 |
| `balance_and_position` | 余额和持仓推送 |
| `liquidation-warning` | 强平预警 |

### 5.5 WebSocket 下单 (Private 通道)

```json
{
  "id": "1234",
  "op": "order",
  "args": [{
    "instId": "ETH-USDT",
    "tdMode": "cash",
    "side": "buy",
    "ordType": "limit",
    "sz": "0.1",
    "px": "2500"
  }]
}
```

支持的 op: `order`, `batch-orders`, `cancel-order`, `batch-cancel-orders`, `amend-order`, `batch-amend-orders`, `mass-cancel`

### 5.6 Python WebSocket 示例

```python
import asyncio
from okx.websocket.WsPublicAsync import WsPublicAsync
from okx.websocket.WsPrivateAsync import WsPrivateAsync

# Public (行情)
async def public_example():
    ws = WsPublicAsync(url="wss://ws.okx.com:8443/ws/v5/public")
    await ws.start()
    await ws.subscribe(
        [{"channel": "trades", "instId": "ETH-USDT"}],
        callback=lambda msg: print(msg)
    )
    await asyncio.sleep(30)
    await ws.stop()

# Private (订单推送)
async def private_example():
    ws = WsPrivateAsync(
        apiKey="xxx", passphrase="xxx", secretKey="xxx",
        url="wss://ws.okx.com:8443/ws/v5/private"
    )
    await ws.start()
    await ws.subscribe(
        [{"channel": "orders", "instType": "SPOT"}],
        callback=lambda msg: print(msg)
    )
    await asyncio.sleep(30)
    await ws.stop()
```

## 6. 产品类型 (instType)

| 值 | 说明 | instId 示例 |
|----|------|------------|
| `SPOT` | 现货 | ETH-USDT |
| `MARGIN` | 杠杆 | ETH-USDT |
| `SWAP` | 永续合约 | ETH-USDT-SWAP |
| `FUTURES` | 交割合约 | ETH-USDT-240329 |
| `OPTION` | 期权 | ETH-USD-240329-2500-C |

## 7. 交易模式 (tdMode)

| 值 | 说明 |
|----|------|
| `cash` | 现货非保证金 |
| `cross` | 全仓保证金 |
| `isolated` | 逐仓保证金 |

## 8. 速率限制

REST API 按端点有不同的速率限制，通常：
- 下单: 60次/2s (单产品)
- 批量下单: 300次/2s
- 查询: 20次/2s
- 行情: 20次/2s

WebSocket:
- 下单: 60次/s
- 订阅: 240次/小时

具体限制参考官方文档各端点说明。

## 9. 与本项目交易系统的对接要点

对于当前的 ETH 量化策略，核心需要：

1. **行情获取**: WebSocket 订阅 `trades` + `books5` (ETH-USDT) 实时获取成交和深度
2. **下单**: REST `POST /api/v5/trade/order` 或 WebSocket `op: order`
3. **止盈止损**: 下单时附带 `attachAlgoOrds` 或单独用 `POST /api/v5/trade/order-algo`
4. **持仓查询**: `GET /api/v5/account/positions`
5. **账户余额**: `GET /api/v5/account/balance`
6. **产品信息**: `GET /api/v5/public/instruments?instType=SPOT&instId=ETH-USDT` 获取最小下单量、价格精度等
