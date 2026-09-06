FROM node:22-slim AS runtime

ARG APT_MIRROR=deb.debian.org
RUN if [ "$APT_MIRROR" != "deb.debian.org" ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3 make g++ curl tini ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev --ignore-scripts --legacy-peer-deps --no-audit --no-fund
COPY . .

RUN mkdir -p /data/tdai-memory /data/config
ENV NODE_ENV=production \
    TDAI_GATEWAY_CONFIG=/data/config/tdai-gateway.yaml \
    TDAI_GATEWAY_HOST=0.0.0.0 \
    TDAI_DATA_DIR=/data/tdai-memory \
    NODE_OPTIONS="--max-old-space-size=1536"

EXPOSE 8420
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
  CMD curl -fsS http://127.0.0.1:${TDAI_GATEWAY_PORT:-8420}/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["node", "--import", "tsx", "src/gateway/server.ts"]
