"""Utility for fetching deposit/withdraw fee configuration from multiple exchanges."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PythonFeeFetcher/1.0",
    "Accept": "application/json, text/plain, */*",
}


@dataclass
class NetworkFeeRecord:
    exchange: str
    symbol: str
    network: str
    deposit_enabled: Optional[bool]
    withdraw_enabled: Optional[bool]
    deposit_fixed_fee: Optional[float]
    deposit_percent_fee: Optional[float]
    withdraw_fixed_fee: Optional[float]
    withdraw_percent_fee: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HttpError(RuntimeError):
    pass


def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"
    request_headers = dict(DEFAULT_HEADERS)
    if headers:
        request_headers.update(headers)
    req = Request(url, headers=request_headers)
    try:
        with urlopen(req, timeout=20) as resp:
            payload = resp.read().decode("utf-8")
            return json.loads(payload)
    except Exception as exc:  # pragma: no cover - network failures are expected to bubble up
        raise HttpError(f"Failed to fetch {url}: {exc}") from exc


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "enable", "enabled", "open", "normal"}:
            return True
        if lowered in {"false", "0", "no", "disable", "disabled", "close", "closed"}:
            return False
    return None


def _to_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


class BaseFetcher:
    name: str

    def fetch(self) -> List[NetworkFeeRecord]:
        raise NotImplementedError

    def _log_fetch(self, count: int) -> None:
        logger.info("%s: fetched %s records", self.name, count)


class GateFetcher(BaseFetcher):
    name = "gate"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://api.gateio.ws/api/v4/spot/currencies")
        records: List[NetworkFeeRecord] = []
        for currency in data:
            symbol = currency.get("currency")
            chains: Iterable[Dict[str, Any]] = currency.get("chains") or []
            for chain in chains:
                network = chain.get("name") or currency.get("chain") or ""
                records.append(
                    NetworkFeeRecord(
                        exchange=self.name,
                        symbol=symbol,
                        network=network,
                        deposit_enabled=_to_optional_bool(not chain.get("deposit_disabled")),
                        withdraw_enabled=_to_optional_bool(not chain.get("withdraw_disabled")),
                        deposit_fixed_fee=None,
                        deposit_percent_fee=None,
                        withdraw_fixed_fee=None,
                        withdraw_percent_fee=None,
                    )
                )
        self._log_fetch(len(records))
        return records


class OkxFetcher(BaseFetcher):
    name = "okx"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://www.okx.com/api/v5/asset/currencies")
        payload: Iterable[Dict[str, Any]] = data.get("data", [])
        records: List[NetworkFeeRecord] = []
        for entry in payload:
            symbol = entry.get("ccy")
            chain = entry.get("chain") or entry.get("chainType")
            records.append(
                NetworkFeeRecord(
                    exchange=self.name,
                    symbol=symbol,
                    network=chain or "",
                    deposit_enabled=_to_optional_bool(entry.get("canDep")),
                    withdraw_enabled=_to_optional_bool(entry.get("canWd")),
                    deposit_fixed_fee=_to_optional_float(entry.get("depQuotaFixed")),
                    deposit_percent_fee=_to_optional_float(entry.get("depQuotaPct")),
                    withdraw_fixed_fee=_to_optional_float(entry.get("minFee")),
                    withdraw_percent_fee=_to_optional_float(entry.get("feeRate")),
                )
            )
        self._log_fetch(len(records))
        return records


class HtxFetcher(BaseFetcher):
    name = "htx"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://api.huobi.pro/v2/reference/currencies")
        payload: Iterable[Dict[str, Any]] = data.get("data", [])
        records: List[NetworkFeeRecord] = []
        for entry in payload:
            symbol = entry.get("currency")
            chains: Iterable[Dict[str, Any]] = entry.get("chains") or []
            for chain in chains:
                records.append(
                    NetworkFeeRecord(
                        exchange=self.name,
                        symbol=symbol,
                        network=chain.get("chain") or chain.get("displayName") or "",
                        deposit_enabled=_to_optional_bool(chain.get("depositStatus")),
                        withdraw_enabled=_to_optional_bool(chain.get("withdrawStatus")),
                        deposit_fixed_fee=_to_optional_float(chain.get("transactFeeDeposit")),
                        deposit_percent_fee=_to_optional_float(chain.get("transactFeeDepositPercent")),
                        withdraw_fixed_fee=_to_optional_float(chain.get("transactFeeWithdraw")),
                        withdraw_percent_fee=_to_optional_float(chain.get("transactFeeWithdrawPercent")),
                    )
                )
        self._log_fetch(len(records))
        return records


class BitgetFetcher(BaseFetcher):
    name = "bitget"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://api.bitget.com/api/spot/v1/public/currencies")
        payload: Iterable[Dict[str, Any]] = data.get("data", [])
        records: List[NetworkFeeRecord] = []
        for entry in payload:
            symbol = entry.get("coinName") or entry.get("coinId")
            for chain in entry.get("chains", []) or []:
                records.append(
                    NetworkFeeRecord(
                        exchange=self.name,
                        symbol=symbol,
                        network=chain.get("chain") or chain.get("shortName") or "",
                        deposit_enabled=_to_optional_bool(chain.get("rechargeable")),
                        withdraw_enabled=_to_optional_bool(chain.get("withdrawable")),
                        deposit_fixed_fee=_to_optional_float(chain.get("rechargeFee")),
                        deposit_percent_fee=_to_optional_float(chain.get("rechargeFeeRate")),
                        withdraw_fixed_fee=_to_optional_float(chain.get("withdrawFee")),
                        withdraw_percent_fee=_to_optional_float(chain.get("withdrawFeeRate")),
                    )
                )
        self._log_fetch(len(records))
        return records


class BybitFetcher(BaseFetcher):
    name = "bybit"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://api.bybit.com/asset/v1/public/coins")
        payload = data.get("result", {}).get("rows", [])
        records: List[NetworkFeeRecord] = []
        for entry in payload:
            symbol = entry.get("coin") or entry.get("coinName")
            for chain in entry.get("chains", []) or []:
                records.append(
                    NetworkFeeRecord(
                        exchange=self.name,
                        symbol=symbol,
                        network=chain.get("chainType") or chain.get("chain") or "",
                        deposit_enabled=_to_optional_bool(chain.get("depositable")),
                        withdraw_enabled=_to_optional_bool(chain.get("withdrawable")),
                        deposit_fixed_fee=_to_optional_float(chain.get("depositFee")),
                        deposit_percent_fee=_to_optional_float(chain.get("depositFeeRate")),
                        withdraw_fixed_fee=_to_optional_float(chain.get("withdrawFee")),
                        withdraw_percent_fee=_to_optional_float(chain.get("withdrawFeeRate")),
                    )
                )
        self._log_fetch(len(records))
        return records


class MexcFetcher(BaseFetcher):
    name = "mexc"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://www.mexc.com/open/api/v3/market/coin/list")
        payload: Iterable[Dict[str, Any]] = data.get("data", [])
        records: List[NetworkFeeRecord] = []
        for entry in payload:
            symbol = entry.get("currency") or entry.get("symbol")
            for chain in entry.get("coins", []) or []:
                records.append(
                    NetworkFeeRecord(
                        exchange=self.name,
                        symbol=symbol,
                        network=chain.get("network") or chain.get("name") or "",
                        deposit_enabled=_to_optional_bool(chain.get("enableDeposit")),
                        withdraw_enabled=_to_optional_bool(chain.get("enableWithdraw")),
                        deposit_fixed_fee=_to_optional_float(chain.get("depositFee")),
                        deposit_percent_fee=_to_optional_float(chain.get("depositFeeRate")),
                        withdraw_fixed_fee=_to_optional_float(chain.get("withdrawFee")),
                        withdraw_percent_fee=_to_optional_float(chain.get("withdrawFeeRate")),
                    )
                )
        self._log_fetch(len(records))
        return records


class KucoinFetcher(BaseFetcher):
    name = "kucoin"

    def fetch(self) -> List[NetworkFeeRecord]:
        data = _http_get_json("https://api.kucoin.com/api/v3/currencies")
        payload: Iterable[Dict[str, Any]] = data.get("data", [])
        records: List[NetworkFeeRecord] = []
        for entry in payload:
            symbol = entry.get("currency")
            for chain in entry.get("chains", []) or []:
                records.append(
                    NetworkFeeRecord(
                        exchange=self.name,
                        symbol=symbol,
                        network=chain.get("chainName") or chain.get("chainId") or "",
                        deposit_enabled=_to_optional_bool(chain.get("isDepositEnabled")),
                        withdraw_enabled=_to_optional_bool(chain.get("isWithdrawEnabled")),
                        deposit_fixed_fee=_to_optional_float(chain.get("depositFee")),
                        deposit_percent_fee=_to_optional_float(chain.get("depositFeeRate")),
                        withdraw_fixed_fee=_to_optional_float(chain.get("withdrawalMinFee")),
                        withdraw_percent_fee=_to_optional_float(chain.get("withdrawFeeRate")),
                    )
                )
        self._log_fetch(len(records))
        return records


FETCHERS: List[BaseFetcher] = [
    GateFetcher(),
    OkxFetcher(),
    HtxFetcher(),
    BitgetFetcher(),
    BybitFetcher(),
    MexcFetcher(),
    KucoinFetcher(),
]


def fetch_all() -> List[NetworkFeeRecord]:
    all_records: List[NetworkFeeRecord] = []
    for fetcher in FETCHERS:
        started = time.time()
        try:
            records = fetcher.fetch()
            all_records.extend(records)
        except HttpError as exc:
            logger.error("%s: %s", fetcher.name, exc)
        finally:
            logger.debug("%s: completed in %.2fs", fetcher.name, time.time() - started)
    return all_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    records = fetch_all()
    logger.info("Total standardized entries: %s", len(records))
    sample = [record.to_dict() for record in records[:5]]
    print(json.dumps(sample, ensure_ascii=False, indent=2))
