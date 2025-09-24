"""Fetch standardized coin and network fee information from multiple exchanges.

This module exposes helper functions to download deposit/withdrawal availability,
fixed withdrawal fees and percentage-based withdrawal fees (when available)
from a set of centralized exchanges. The structure of responses differs from
exchange to exchange, so each exchange has a dedicated parser that normalizes
its payload into a shared schema.

The resulting schema for every entry looks like the following:

```
{
    "exchange": "gate",
    "asset": "USDT",
    "network": "TRX",
    "deposit_enabled": true,
    "withdraw_enabled": true,
    "withdraw_fee_fixed": "1",
    "withdraw_fee_percent": null
}
```

Only publicly accessible REST endpoints are used. Some exchanges (for example
OKX, Bybit or MEXC) might apply geo/IP filtering or require API keys for
selected endpoints. In such cases the fetcher will raise a descriptive error
that can be inspected by callers.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FeeFetcher/1.0; +https://openai.com/)",
    "Accept": "application/json",
}


@dataclass
class NetworkFeeRecord:
    """Standardized representation of a single network configuration."""

    exchange: str
    asset: str
    network: Optional[str]
    deposit_enabled: Optional[bool]
    withdraw_enabled: Optional[bool]
    withdraw_fee_fixed: Optional[str]
    withdraw_fee_percent: Optional[str]

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to plain dictionary, omitting type metadata."""

        return asdict(self)


def _fetch_json(url: str, params: Optional[Dict[str, str]] = None, headers: Optional[Dict[str, str]] = None) -> dict:
    """Download a JSON payload from *url*.

    Args:
        url: Base URL.
        params: Optional dictionary of query parameters appended to the URL.
        headers: Optional HTTP headers. Defaults to :data:`DEFAULT_HEADERS`.

    Returns:
        Parsed JSON document (converted to native Python objects).

    Raises:
        urllib.error.HTTPError: Propagated when the HTTP request fails.
        json.JSONDecodeError: If the response is not valid JSON.
    """

    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"

    request = urllib.request.Request(url, headers=headers or DEFAULT_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    return json.loads(payload)


def _safe_bool(value: Optional[str | bool]) -> Optional[bool]:
    """Normalize truthy values expressed as booleans or strings."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "allowed", "enable", "enabled"}:
        return True
    if text in {"0", "false", "f", "no", "n", "disabled", "forbidden", "ban", "prohibited", "suspend", "suspended", "halt"}:
        return False
    return None


def _safe_decimal(value: Optional[str]) -> Optional[str]:
    """Convert a numeric value to a canonical decimal string.

    The API responses use various numeric formats (strings, ints, floats). To
    avoid floating point precision loss, everything is converted through
    :class:`~decimal.Decimal` and serialized back to a string.
    """

    if value in (None, "", "null"):
        return None
    try:
        return format(Decimal(str(value)), "f")
    except (InvalidOperation, ValueError):
        return None


def _record(exchange: str, asset: str, network: Optional[str], deposit: Optional[bool], withdraw: Optional[bool], fixed_fee: Optional[str], percent_fee: Optional[str]) -> NetworkFeeRecord:
    """Small helper to construct :class:`NetworkFeeRecord` instances."""

    return NetworkFeeRecord(
        exchange=exchange,
        asset=asset,
        network=network,
        deposit_enabled=deposit,
        withdraw_enabled=withdraw,
        withdraw_fee_fixed=fixed_fee,
        withdraw_fee_percent=percent_fee,
    )


def fetch_gate() -> List[NetworkFeeRecord]:
    """Fetch currency metadata from Gate.io."""

    url = "https://api.gateio.ws/api/v4/spot/currencies"
    payload = _fetch_json(url)
    records: List[NetworkFeeRecord] = []
    for item in payload:
        asset = item.get("currency")
        chains = item.get("chains") or []
        if not chains:
            deposit_disabled = _safe_bool(item.get("deposit_disabled"))
            withdraw_disabled = _safe_bool(item.get("withdraw_disabled"))
            records.append(
                _record(
                    exchange="gate",
                    asset=asset,
                    network=item.get("chain"),
                    deposit=None if deposit_disabled is None else not deposit_disabled,
                    withdraw=None if withdraw_disabled is None else not withdraw_disabled,
                    fixed_fee=None,
                    percent_fee=None,
                )
            )
            continue
        for chain in chains:
            deposit_disabled = _safe_bool(chain.get("deposit_disabled"))
            withdraw_disabled = _safe_bool(chain.get("withdraw_disabled"))
            deposit = None if deposit_disabled is None else not deposit_disabled
            withdraw = None if withdraw_disabled is None else not withdraw_disabled
            records.append(
                _record(
                    exchange="gate",
                    asset=asset,
                    network=chain.get("name"),
                    deposit=deposit,
                    withdraw=withdraw,
                    fixed_fee=_safe_decimal(chain.get("withdraw_fixed_fee")),
                    percent_fee=_safe_decimal(chain.get("withdraw_percent_fee")),
                )
            )
    return records


def fetch_okx() -> List[NetworkFeeRecord]:
    """Fetch currency metadata from OKX.

    OKX enforces geo restrictions for some regions. When a 4xx/5xx error is
    raised the caller should inspect the exception message.
    """

    url = "https://www.okx.com/api/v5/asset/currencies"
    payload = _fetch_json(url)
    records: List[NetworkFeeRecord] = []
    for item in payload.get("data", []):
        asset = item.get("ccy")
        for chain in item.get("chains", []):
            records.append(
                _record(
                    exchange="okx",
                    asset=asset,
                    network=chain.get("chain"),
                    deposit=_safe_bool(chain.get("canDep")),
                    withdraw=_safe_bool(chain.get("canWd")),
                    fixed_fee=_safe_decimal(chain.get("minFee")),
                    percent_fee=_safe_decimal(chain.get("feeRate")),
                )
            )
    return records


def fetch_htx() -> List[NetworkFeeRecord]:
    """Fetch currency metadata from HTX (Huobi)."""

    url = "https://api.huobi.pro/v2/reference/currencies"
    payload = _fetch_json(url)
    records: List[NetworkFeeRecord] = []
    for item in payload.get("data", []):
        asset = item.get("currency")
        for chain in item.get("chains", []):
            withdraw_fee_type = chain.get("withdrawFeeType")
            percent_fee = None
            fixed_fee = None
            if withdraw_fee_type == "ratio":
                percent_fee = _safe_decimal(chain.get("transactFeeRateWithdraw") or chain.get("withdrawFeeRate"))
                fixed_fee = _safe_decimal(chain.get("minTransactFeeWithdraw"))
            else:
                fixed_fee = _safe_decimal(chain.get("transactFeeWithdraw"))
                percent_fee = _safe_decimal(chain.get("withdrawFeeRate"))
            records.append(
                _record(
                    exchange="htx",
                    asset=asset,
                    network=chain.get("chain"),
                    deposit=_safe_bool(chain.get("depositStatus")),
                    withdraw=_safe_bool(chain.get("withdrawStatus")),
                    fixed_fee=fixed_fee,
                    percent_fee=percent_fee,
                )
            )
    return records


def fetch_bitget() -> List[NetworkFeeRecord]:
    """Fetch currency metadata from Bitget."""

    url = "https://api.bitget.com/api/spot/v1/public/currencies"
    payload = _fetch_json(url)
    records: List[NetworkFeeRecord] = []
    for item in payload.get("data", []):
        asset = item.get("coinName")
        for chain in item.get("chains", []):
            records.append(
                _record(
                    exchange="bitget",
                    asset=asset,
                    network=chain.get("chain"),
                    deposit=_safe_bool(chain.get("rechargeable")),
                    withdraw=_safe_bool(chain.get("withdrawable")),
                    fixed_fee=_safe_decimal(chain.get("withdrawFee")),
                    percent_fee=_safe_decimal(chain.get("extraWithDrawFee")),
                )
            )
    return records


def fetch_bybit() -> List[NetworkFeeRecord]:
    """Fetch currency metadata from Bybit."""

    url = "https://api.bybit.com/v5/asset/coin/query-info"
    payload = _fetch_json(url)
    records: List[NetworkFeeRecord] = []
    result = payload.get("result", {})
    for row in result.get("rows", []):
        asset = row.get("coin")
        for chain in row.get("chains", []):
            records.append(
                _record(
                    exchange="bybit",
                    asset=asset,
                    network=chain.get("chain"),
                    deposit=_safe_bool(chain.get("chainDeposit")),
                    withdraw=_safe_bool(chain.get("chainWithdraw")),
                    fixed_fee=_safe_decimal(chain.get("withdrawFee")),
                    percent_fee=_safe_decimal(chain.get("withdrawFeeRate")),
                )
            )
    return records


def fetch_mexc() -> List[NetworkFeeRecord]:
    """Fetch currency metadata from MEXC."""

    url = "https://api.mexc.com/api/v3/capital/config/getall"
    payload = _fetch_json(url)
    records: List[NetworkFeeRecord] = []
    for item in payload:
        asset = item.get("coin")
        for chain in item.get("networkList", []):
            records.append(
                _record(
                    exchange="mexc",
                    asset=asset,
                    network=chain.get("network"),
                    deposit=_safe_bool(chain.get("depositEnable")),
                    withdraw=_safe_bool(chain.get("withdrawEnable")),
                    fixed_fee=_safe_decimal(chain.get("withdrawFee")),
                    percent_fee=_safe_decimal(chain.get("withdrawFeeRate")),
                )
            )
    return records


def fetch_kucoin(delay: float = 0.2) -> List[NetworkFeeRecord]:
    """Fetch currency metadata from KuCoin.

    KuCoin splits metadata across two endpoints. The first endpoint returns a
    list of available assets, and the second exposes per-network details.
    A small delay between requests is recommended to avoid hitting the public
    rate limits.
    """

    base_url = "https://api.kucoin.com"
    summary = _fetch_json(f"{base_url}/api/v1/currencies")
    records: List[NetworkFeeRecord] = []
    for item in summary.get("data", []):
        asset = item.get("currency")
        try:
            detail = _fetch_json(f"{base_url}/api/v2/currencies/{asset}")
        except urllib.error.HTTPError as exc:  # pragma: no cover - passthrough network errors
            # Fallback to the limited summary payload when the detail endpoint
            # is unavailable for a particular asset.
            records.append(
                _record(
                    exchange="kucoin",
                    asset=asset,
                    network=None,
                    deposit=_safe_bool(item.get("isDepositEnabled")),
                    withdraw=_safe_bool(item.get("isWithdrawEnabled")),
                    fixed_fee=_safe_decimal(item.get("withdrawalMinFee")),
                    percent_fee=None,
                )
            )
            continue

        for chain in detail.get("data", {}).get("chains", []):
            records.append(
                _record(
                    exchange="kucoin",
                    asset=asset,
                    network=chain.get("chain"),
                    deposit=_safe_bool(chain.get("isDepositEnabled")),
                    withdraw=_safe_bool(chain.get("isWithdrawEnabled")),
                    fixed_fee=_safe_decimal(chain.get("withdrawalMinFee")),
                    percent_fee=_safe_decimal(chain.get("withdrawFeeRate")),
                )
            )
        time.sleep(delay)
    return records


EXCHANGE_FETCHERS = {
    "gate": fetch_gate,
    "okx": fetch_okx,
    "htx": fetch_htx,
    "bitget": fetch_bitget,
    "bybit": fetch_bybit,
    "mexc": fetch_mexc,
    "kucoin": fetch_kucoin,
}


def fetch_all(exchanges: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Fetch and normalize withdrawal/deposit information for *exchanges*.

    Args:
        exchanges: Optional iterable of exchange identifiers. When omitted all
            supported exchanges are queried.

    Returns:
        Dictionary keyed by exchange name. Each value is a list of normalized
        dictionaries ready for serialization.
    """

    selected = list(exchanges) if exchanges else list(EXCHANGE_FETCHERS.keys())
    result: Dict[str, Any] = {}
    for name in selected:
        fetcher = EXCHANGE_FETCHERS.get(name.lower())
        if not fetcher:
            raise KeyError(f"Unsupported exchange: {name}")
        try:
            records = fetcher()
        except Exception as exc:  # pragma: no cover - passthrough network errors
            result[name.lower()] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        result[name.lower()] = [record.to_dict() for record in records]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch standardized exchange fee data.")
    parser.add_argument(
        "--exchange",
        "-e",
        dest="exchanges",
        action="append",
        help="Fetch a specific exchange (can be supplied multiple times). Defaults to all exchanges.",
    )
    parser.add_argument(
        "--sleep",
        dest="kucoin_delay",
        type=float,
        default=0.2,
        help="Delay between KuCoin detail requests in seconds (default: 0.2).",
    )
    args = parser.parse_args()

    if args.exchanges:
        # Update KuCoin delay when the fetcher is instantiated via the helper map.
        selected = [name.lower() for name in args.exchanges]
        if "kucoin" in selected:
            EXCHANGE_FETCHERS["kucoin"] = lambda delay=args.kucoin_delay: fetch_kucoin(delay=delay)
        data = fetch_all(selected)
    else:
        # Override KuCoin fetcher with the provided delay for the default flow.
        EXCHANGE_FETCHERS["kucoin"] = lambda delay=args.kucoin_delay: fetch_kucoin(delay=delay)
        data = fetch_all()

    print(json.dumps(data, ensure_ascii=False, indent=2))
