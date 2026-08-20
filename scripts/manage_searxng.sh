#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy/searxng"
COMPOSE_FILE="$DEPLOY_DIR/compose.yaml"
ENV_FILE="$DEPLOY_DIR/.env"

usage() {
    echo "Usage: $0 {up|stop|status|test|logs}" >&2
}

ensure_runtime_env() {
    if [[ -f "$ENV_FILE" ]]; then
        return
    fi

    local secret
    secret="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    (
        umask 077
        printf 'SEARXNG_SECRET=%s\n' "$secret" > "$ENV_FILE"
    )
    echo "Created private runtime configuration: $ENV_FILE"
}

compose() {
    docker compose \
        --file "$COMPOSE_FILE" \
        --env-file "$ENV_FILE" \
        "$@"
}

require_runtime() {
    command -v docker >/dev/null 2>&1 || {
        echo "docker is required" >&2
        exit 1
    }
    command -v curl >/dev/null 2>&1 || {
        echo "curl is required" >&2
        exit 1
    }
    docker compose version >/dev/null
    ensure_runtime_env
}

wait_until_ready() {
    local attempt
    for attempt in {1..30}; do
        if curl --fail --silent --output /dev/null \
            "http://127.0.0.1:8080/healthz"; then
            echo "SearXNG is ready on http://127.0.0.1:8080"
            return
        fi
        sleep 1
    done

    echo "SearXNG did not become ready within 30 seconds" >&2
    compose logs --tail 100 searxng >&2
    exit 1
}

command="${1:-}"
case "$command" in
    up)
        require_runtime
        compose up --detach --pull missing
        wait_until_ready
        ;;
    stop)
        require_runtime
        compose stop
        ;;
    status)
        require_runtime
        compose ps
        ;;
    test)
        require_runtime
        curl --fail --silent --show-error --get \
            --data-urlencode "q=今天的人工智能新闻" \
            --data-urlencode "format=json" \
            --data-urlencode "language=zh-CN" \
            "http://127.0.0.1:8080/search" \
            | python3 -c 'import json, sys; data=json.load(sys.stdin); results=data.get("results", []); assert isinstance(results, list); print(f"SearXNG JSON API OK: {len(results)} results")'
        ;;
    logs)
        require_runtime
        compose logs --tail 100 searxng
        ;;
    *)
        usage
        exit 2
        ;;
esac
