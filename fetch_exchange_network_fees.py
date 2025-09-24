"""Utility for fetching standardized deposit/withdraw fee information from major exchanges.

The script relies on the ccxt library to talk to exchange REST APIs and returns
all data in a normalized JSON structure without doing any aggregation or post-processing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ccxt  # type: ignore


@dataclass
class NetworkFeeRecord:
    """Normalized representation of the deposit/withdraw status for a network."""

    exchange: str
    asset: str
    network: str
    deposit_enabled: Optional[bool]
    withdraw_enabled: Optional[bool]
    withdraw_fee_fixed: Optional[float]
    withdraw_fee_percent: Optional[float]
    deposit_fee_fixed: Optional[float]
    deposit_fee_percent: Optional[float]
    raw_network: Dict[str, Any]
    raw_currency: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert enums/complex objects to pure python structures for JSON serialization.
        return result


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "enabled", "open"}:
            return True
        if lowered in {"false", "f", "0", "disabled", "close", "closed"}:
            return False
    return None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_network_records(
    exchange_id: str,
    asset_code: str,
    currency_payload: Dict[str, Any],
) -> Iterable[NetworkFeeRecord]:
    networks = currency_payload.get("networks") or {}
    if not networks:
        # When the exchange does not break data down per network we still emit a single entry.
        yield NetworkFeeRecord(
            exchange=exchange_id,
            asset=asset_code,
            network=currency_payload.get("network") or "_DEFAULT",
            deposit_enabled=_to_bool(currency_payload.get("deposit")),
            withdraw_enabled=_to_bool(currency_payload.get("withdraw")),
            withdraw_fee_fixed=_to_float(currency_payload.get("fee") or currency_payload.get("withdrawFee")),
            withdraw_fee_percent=_to_float(currency_payload.get("percentage")),
            deposit_fee_fixed=_to_float(currency_payload.get("depositFee")),
            deposit_fee_percent=_to_float(currency_payload.get("depositFeePercent")),
            raw_network={},
            raw_currency=currency_payload,
        )
        return

    for network_name, network_payload in networks.items():
        if isinstance(network_payload, dict):
            deposit_enabled = _to_bool(network_payload.get("deposit"))
            withdraw_enabled = _to_bool(network_payload.get("withdraw"))
            withdraw_fee_fixed = _to_float(
                network_payload.get("fee")
                or network_payload.get("withdrawFee")
                or network_payload.get("withdraw_fee")
            )
            withdraw_fee_percent = _to_float(
                network_payload.get("percentage")
                or network_payload.get("withdrawPercentage")
            )
            deposit_fee_fixed = _to_float(
                network_payload.get("depositFee")
                or network_payload.get("deposit_fee")
            )
            deposit_fee_percent = _to_float(network_payload.get("depositFeePercent"))
        else:
            # Some exchanges use arrays instead of dicts for networks.
            deposit_enabled = None
            withdraw_enabled = None
            withdraw_fee_fixed = None
            withdraw_fee_percent = None
            deposit_fee_fixed = None
            deposit_fee_percent = None

        yield NetworkFeeRecord(
            exchange=exchange_id,
            asset=asset_code,
            network=str(network_name),
            deposit_enabled=deposit_enabled,
            withdraw_enabled=withdraw_enabled,
            withdraw_fee_fixed=withdraw_fee_fixed,
            withdraw_fee_percent=withdraw_fee_percent,
            deposit_fee_fixed=deposit_fee_fixed,
            deposit_fee_percent=deposit_fee_percent,
            raw_network=network_payload if isinstance(network_payload, dict) else {"value": network_payload},
            raw_currency=currency_payload,
        )


def _instantiate_exchange(exchange_id: str, credential_prefix: str) -> ccxt.Exchange:
    klass = getattr(ccxt, exchange_id)
    api_key = os.getenv(f"{credential_prefix}_API_KEY")
    secret = os.getenv(f"{credential_prefix}_API_SECRET")
    password = os.getenv(f"{credential_prefix}_API_PASSWORD") or os.getenv(f"{credential_prefix}_API_PASS")
    exchange_kwargs: Dict[str, Any] = {
        "enableRateLimit": True,
        "timeout": 20000,
    }
    if api_key and secret:
        exchange_kwargs.update({"apiKey": api_key, "secret": secret})
    if password:
        exchange_kwargs["password"] = password
    return klass(exchange_kwargs)


def fetch_all_exchange_fees() -> List[NetworkFeeRecord]:
    exchanges: List[Tuple[str, str]] = [
        ("gateio", "GATE"),
        ("okx", "OKX"),
        ("htx", "HTX"),
        ("bitget", "BITGET"),
        ("bybit", "BYBIT"),
        ("mexc", "MEXC"),
        ("kucoin", "KUCOIN"),
    ]

    results: List[NetworkFeeRecord] = []

    for exchange_id, prefix in exchanges:
        exchange = _instantiate_exchange(exchange_id, prefix)
        try:
            currencies = exchange.fetch_currencies()
        except Exception as exc:  # noqa: BLE001 - we want to capture all transport issues
            print(f"Failed to fetch currencies for {exchange_id}: {exc}")
            continue

        if not currencies:
            print(f"Exchange {exchange_id} did not return currency metadata")
            continue

        for asset_code, payload in currencies.items():
            try:
                results.extend(_extract_network_records(exchange_id, asset_code, payload))
            except Exception as exc:  # noqa: BLE001 - data inconsistencies should not abort the whole run
                print(f"Failed to normalize {exchange_id} {asset_code}: {exc}")
                continue

    return results


def main() -> None:
    records = fetch_all_exchange_fees()
    json_payload = [record.to_dict() for record in records]
    print(json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
