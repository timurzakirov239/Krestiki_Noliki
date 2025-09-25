"""Утилита для получения статусов сетей и комиссий на бирже Gate.io."""
import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

API_BASE_URL = "https://api.gateio.ws"
SPOT_CURRENCIES_PATH = "/api/v4/spot/currencies"
WITHDRAW_STATUS_PATH = "/api/v4/wallet/withdraw_status"

# Значения по умолчанию для ключей API, предоставленных пользователем
DEFAULT_API_KEY = "0e88eec4034f9d4b966540d9f61016d9"
DEFAULT_API_SECRET = "2cc5f2c6a6d536b87b711dbd6a7fe9995131f35918c7ff4fdc88ac514d2c7f3a"


def http_get(url: str, *, headers: Optional[Mapping[str, str]] = None, timeout: int = 15) -> object:
    """Вспомогательная функция GET-запроса с обработкой ошибок."""
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="ignore")
        raise RuntimeError(f"HTTP {err.code} при обращении к {url}: {body}") from err
    except urllib.error.URLError as err:  # type: ignore[unreachable]
        raise RuntimeError(f"Ошибка сети при обращении к {url}: {err}") from err


def build_signature(
    method: str,
    path: str,
    query: Optional[Mapping[str, str]],
    body: Optional[object],
    api_secret: str,
    timestamp: str,
) -> str:
    """Построить подпись согласно правилам API Gate."""
    query_string = urllib.parse.urlencode(query or {}, doseq=True)

    if body is None:
        body_str = ""
    elif isinstance(body, str):
        body_str = body
    else:
        body_str = json.dumps(body, separators=(",", ":"))

    payload_hash = hashlib.sha512(body_str.encode("utf-8")).hexdigest()
    sign_payload = "\n".join((method.upper(), path, query_string, payload_hash, timestamp))
    signature = hmac.new(api_secret.encode("utf-8"), sign_payload.encode("utf-8"), hashlib.sha512)
    return signature.hexdigest()


def signed_get(path: str, *, query: Optional[Mapping[str, str]] = None, api_key: str, api_secret: str) -> object:
    """Сделать авторизованный GET-запрос к приватному API."""
    timestamp = str(time.time())
    query_string = urllib.parse.urlencode(query or {}, doseq=True)
    url = f"{API_BASE_URL}{path}"
    if query_string:
        url = f"{url}?{query_string}"

    signature = build_signature("GET", path, query, None, api_secret, timestamp)
    headers = {
        "Accept": "application/json",
        "KEY": api_key,
        "Timestamp": timestamp,
        "SIGN": signature,
    }
    return http_get(url, headers=headers)


def fetch_spot_currencies() -> List[MutableMapping[str, object]]:
    """Получить полный перечень монет и их сетей из публичного API."""
    url = f"{API_BASE_URL}{SPOT_CURRENCIES_PATH}"
    data = http_get(url)
    if not isinstance(data, list):
        raise RuntimeError("Неожиданный ответ от /spot/currencies")
    return data  # type: ignore[return-value]


def fetch_withdraw_status(api_key: str, api_secret: str) -> List[MutableMapping[str, object]]:
    """Получить комиссии и статусы выводов с приватного метода."""
    data = signed_get(WITHDRAW_STATUS_PATH, api_key=api_key, api_secret=api_secret)
    if not isinstance(data, list):
        raise RuntimeError("Неожиданный ответ от /wallet/withdraw_status")
    return data  # type: ignore[return-value]


def merge_currency_data(
    spot_currencies: Iterable[Mapping[str, object]],
    withdraw_statuses: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Объединить информацию о сетях и комиссиях."""
    result: List[Dict[str, object]] = []
    for currency in spot_currencies:
        code = str(currency.get("currency"))
        chains_data = currency.get("chains") or []
        withdraw_info = withdraw_statuses.get(code, {})
        fix_fees: Mapping[str, str] = withdraw_info.get("withdraw_fix_on_chains") or {}
        percent_fees: Mapping[str, str] = withdraw_info.get("withdraw_percent_on_chains") or {}

        merged_networks: List[Dict[str, object]] = []
        if isinstance(chains_data, list):
            for chain in chains_data:
                if not isinstance(chain, Mapping):
                    continue
                chain_name = str(chain.get("name"))
                merged_networks.append(
                    {
                        "chain": chain_name,
                        "deposit_enabled": not bool(chain.get("deposit_disabled", False)),
                        "withdraw_enabled": not bool(chain.get("withdraw_disabled", False)),
                        "withdraw_fix_fee": fix_fees.get(chain_name),
                        "withdraw_percent_fee": percent_fees.get(chain_name),
                    }
                )

        result.append(
            {
                "currency": code,
                "name": currency.get("name"),
                "networks": merged_networks,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=os.getenv("GATE_API_KEY", DEFAULT_API_KEY),
        help="API ключ Gate.io (можно передать через переменную окружения GATE_API_KEY)",
    )
    parser.add_argument(
        "--api-secret",
        dest="api_secret",
        default=os.getenv("GATE_API_SECRET", DEFAULT_API_SECRET),
        help="Секретный ключ Gate.io (можно передать через переменную окружения GATE_API_SECRET)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Путь к файлу для сохранения результата в формате JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа скрипта."""
    args = parse_args()
    if not args.api_key or not args.api_secret:
        print(
            "Необходимы API ключ и секрет. Передайте их через параметры --api-key/--api-secret "
            "или переменные окружения GATE_API_KEY/GATE_API_SECRET.",
            file=sys.stderr,
        )
        sys.exit(1)

    spot = fetch_spot_currencies()
    withdraw_list = fetch_withdraw_status(args.api_key, args.api_secret)
    withdraw_map = {str(item.get("currency")): item for item in withdraw_list}
    merged = merge_currency_data(spot, withdraw_map)

    output_data = json.dumps(merged, ensure_ascii=False, indent=2)
    print(output_data)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_data)


if __name__ == "__main__":
    main()
