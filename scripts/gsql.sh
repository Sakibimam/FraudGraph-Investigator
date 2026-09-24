#!/usr/bin/env bash
# Run a GSQL file (path relative to repo root) inside the local TigerGraph container.
set -euo pipefail
CTX=${TG_DOCKER_CONTEXT:-colima-tg}
docker --context "$CTX" exec -u tigergraph tigergraph bash -lc "/home/tigergraph/tigergraph/app/cmd/gsql -u tigergraph -p tigergraph /home/tigergraph/project/$1"
