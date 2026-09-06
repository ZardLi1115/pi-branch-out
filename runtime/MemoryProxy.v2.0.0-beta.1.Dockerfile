FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 make g++ curl tini ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json package-lock.json ./
RUN node -e "const fs=require('fs');const p=JSON.parse(fs.readFileSync('package.json','utf8'));delete p.optionalDependencies['@context-proxy/cost-guard'];fs.writeFileSync('package.json',JSON.stringify(p,null,2));" && \
    npm install --no-audit --no-fund
COPY . .

RUN mkdir -p /data/tdai-memory-proxy /data/config /app/logs
ENV NODE_ENV=production \
    PROXY_DB_PATH=/data/tdai-memory-proxy/proxy.db \
    NODE_OPTIONS="--max-old-space-size=1536"

EXPOSE 8096
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
  CMD curl -fsS http://127.0.0.1:8096/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "node", "--import", "tsx/esm", "src/index.ts"]
CMD ["--config", "/data/config.yaml"]
