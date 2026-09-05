FROM node:22-bookworm-slim AS builder

ARG PI_VERSION=0.73.1
RUN npm install --global --prefix /opt/pi-runtime --ignore-scripts \
    "@mariozechner/pi-coding-agent@${PI_VERSION}" \
    && cp /usr/local/bin/node /opt/pi-runtime/bin/node \
    && PATH=/opt/pi-runtime/bin:$PATH /opt/pi-runtime/bin/pi --version

FROM scratch AS export
COPY --from=builder /opt/pi-runtime/ /pi-runtime/
