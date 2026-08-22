"""Shared SSRF guard logic.

WHY its own module? app.main validates a submitted ingest source once, at
request time. The actual network fetch (requests.get / git clone) happens
later, in a background job (app.loaders), sometimes seconds to minutes after
submission. DNS is resolved *again* at fetch time — an attacker can point a
hostname at a public IP for the first lookup (passing validation) and rebind
it to an internal address (169.254.169.254, localhost, ...) before the fetch,
landing exactly in the gap between check and use.

Both call sites need the identical check, so it lives in this leaf module
rather than being duplicated (and drifting) or creating an import cycle.
"""

import ipaddress
import socket


class SSRFError(ValueError):
    """A hostname resolved to a private/internal/non-global address."""


def resolve_public_ips(host: str) -> list[str]:
    """Resolve `host` and verify every returned address is publicly routable.

    Raises SSRFError if resolution fails or any resolved address is private,
    loopback, link-local (e.g. cloud metadata at 169.254.169.254), or
    otherwise non-global. An attacker controls DNS, so checking the hostname
    string is useless — only the resolved IP can be trusted.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise SSRFError(f"Cannot resolve host: {host}") from e

    ips = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise SSRFError(
                f"Refusing to fetch private/internal address ({ip}) for host '{host}'"
            )
        ips.append(str(ip))
    return ips
