"""Utilities to fetch standardized deposit and withdrawal fee information.

This module uses the ccxt unified API to retrieve network level
information about the availability of deposits and withdrawals as well
as fixed and percentage fees for several major crypto exchanges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import ccxt


EXCHANGE_IDS = {
    "gate": "gateio",
    "okx": "okx",
    "htx": "htx",
    "bitget": "bitget",
    "bybit": "bybit",
    "mexc": "mexc",
    "kucoin": "kucoin",
}


@dataclass
class NetworkFeeInfo:
    """Standardized representation of per-network fee information."""

    exchange: str
    currency: str
    network: str
    deposit_enabled: Optional[bool]
    withdraw_enabled: Optional[bool]
    deposit_fee_fixed: Optional[float]
    deposit_fee_percent: Optional[float]
    withdraw_fee_fixed: Optional[float]
    withdraw_fee_percent: Optional[float]
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a plain dictionary for serialization."""

        data = asdict(self)
        data["raw"] = self.raw
        return data


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "yes", "1", "enabled", "enable"}:
            return True
        if lower in {"false", "no", "0", "disabled", "disable"}:
            return False
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", False):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_percentage(info: Any) -> Optional[float]:
    if isinstance(info, dict):
        percentage = info.get("percentage")
        if percentage is not None:
            return _safe_float(percentage)
    else:
        return _safe_float(info)
    return None


def _extract_fee(info: Any) -> Optional[float]:
    if isinstance(info, (int, float, str)):
        return _safe_float(info)
    if not isinstance(info, dict):
        return None
    if "fee" in info:
        return _safe_float(info["fee"])
    limits = info.get("limits")
    if isinstance(limits, dict):
        for limit_key in ("withdraw", "deposit"):
            limit = limits.get(limit_key)
            if isinstance(limit, dict) and "fee" in limit:
                fee_value = _safe_float(limit["fee"])
                if fee_value is not None:
                    return fee_value
    return None


def _extract_status(value: Any, fallback: Any = None) -> Optional[bool]:
    if isinstance(value, dict):
        for key in ("enabled", "enable", "active", "status", "available"):
            if key in value:
                status = _safe_bool(value[key])
                if status is not None:
                    return status
        return _safe_bool(fallback)
    return _safe_bool(value if value is not None else fallback)


def _ensure_dict(value: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return default


def _standardize_network_data(
    exchange_id: str,
    currency_code: str,
    network_code: str,
    network_info: Dict[str, Any],
    currency_info: Optional[Dict[str, Any]] = None,
) -> NetworkFeeInfo:
    currency_info = currency_info or network_info

    deposit_status = _extract_status(
        network_info.get("deposit"),
        fallback=_extract_status(currency_info.get("deposit"), currency_info.get("active")),
    )
    withdraw_status = _extract_status(
        network_info.get("withdraw"),
        fallback=_extract_status(currency_info.get("withdraw"), currency_info.get("active")),
    )

    deposit_container = _ensure_dict(
        network_info.get("deposit"),
        _ensure_dict(currency_info.get("deposit"), currency_info),
    )
    withdraw_container = _ensure_dict(
        network_info.get("withdraw"),
        _ensure_dict(currency_info.get("withdraw"), currency_info),
    )

    deposit_fee_fixed = _extract_fee(deposit_container)
    withdraw_fee_fixed = _extract_fee(withdraw_container)

    deposit_fee_percent = _extract_percentage(deposit_container)
    withdraw_fee_percent = _extract_percentage(withdraw_container)

    return NetworkFeeInfo(
        exchange=exchange_id,
        currency=currency_code,
        network=network_code or (network_info.get("network") or ""),
        deposit_enabled=deposit_status,
        withdraw_enabled=withdraw_status,
        deposit_fee_fixed=deposit_fee_fixed,
        deposit_fee_percent=deposit_fee_percent,
        withdraw_fee_fixed=withdraw_fee_fixed,
        withdraw_fee_percent=withdraw_fee_percent,
        raw=network_info,
    )


def fetch_exchange_network_fees(exchange_id: str) -> List[NetworkFeeInfo]:
    """Fetch standardized network fee data for a single exchange."""

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    currencies = exchange.fetch_currencies()

    standardized: List[NetworkFeeInfo] = []
    for currency_code, currency_info in currencies.items():
        networks = currency_info.get("networks") or {}
        if not networks:
            standardized.append(
                _standardize_network_data(
                    exchange_id,
                    currency_code,
                    currency_info.get("network", ""),
                    currency_info,
                    currency_info,
                )
            )
            continue

        for network_code, network_info in networks.items():
            standardized.append(
                _standardize_network_data(
                    exchange_id,
                    currency_code,
                    network_code,
                    network_info,
                    currency_info,
                )
            )

    return standardized


def fetch_all_exchanges(exchange_aliases: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Fetch network fee data for all configured exchanges."""

    if exchange_aliases is None:
        exchange_aliases = EXCHANGE_IDS.keys()

    all_entries: List[Dict[str, Any]] = []
    for alias in exchange_aliases:
        ccxt_id = EXCHANGE_IDS[alias]
        entries = fetch_exchange_network_fees(ccxt_id)
        all_entries.extend(entry.to_dict() for entry in entries)
    return all_entries


def main() -> None:
    """Fetch and print standardized fee data for all exchanges."""

    data = fetch_all_exchanges()
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
