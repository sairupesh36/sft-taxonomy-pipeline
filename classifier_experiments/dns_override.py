"""
Temporary DNS workaround, this process only. Never touches /etc/hosts, any
shared config, or the actual sglang deployment/infra -- it's a client-side
socket.getaddrinfo monkeypatch that only affects name resolution inside
whichever Python process imports this module.

Root cause: this shell can resolve the k8s API server (kubectl works) but
not the sglang-router service's ClusterIP or DNS name (both time out) --
confirmed to be this shell's own network path, not an actual outage (the
router and gemma-4-31b pods are both Running and healthy per `kubectl get
pods -n sglang`, and respond correctly over the NodePort on any real node
IP, e.g. 172.17.99.1:30000). This lets the existing, unmodified pipeline
scripts (which all hardcode the sglang-router hostname) run as-is by
resolving that one hostname locally to a working node IP instead.

Usage: `import dns_override` before any code that connects to
sglang-router.sglang.svc.cluster.local -- must be imported before aiohttp
opens its connector/resolver.
"""
import socket

_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_OVERRIDE_HOST = "sglang-router.sglang.svc.cluster.local"
_REDIRECT_TO_IP = "172.17.99.1"  # a real cluster node IP; NodePort 30000 works from here


def _patched_getaddrinfo(host, *args, **kwargs):
    if host == _OVERRIDE_HOST:
        host = _REDIRECT_TO_IP
    return _ORIGINAL_GETADDRINFO(host, *args, **kwargs)


socket.getaddrinfo = _patched_getaddrinfo
print(f"[dns_override] {_OVERRIDE_HOST} -> {_REDIRECT_TO_IP} (this process only)")
