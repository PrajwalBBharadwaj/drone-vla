#!/usr/bin/env python3
"""Wait for a local HTTP service before systemd starts its dependants."""
import argparse
import sys
import time
import urllib.request


def wait_http(url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as reply:
                if reply.status == 200:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Service did not become ready: {url}. Inspect sudo journalctl -u ollama -u drone-vla -n 100.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        wait_http(args.url, args.timeout)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
