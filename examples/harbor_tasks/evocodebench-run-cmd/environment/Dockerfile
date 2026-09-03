# DaoCloud + Aliyun mirrors: Docker Hub / archive.ubuntu.com are often
# unreachable from the network this example was first run on. Swap back to
# `ubuntu:24.04` and default apt sources if your network can reach them.
FROM docker.m.daocloud.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list 2>/dev/null || true \
    && if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
         sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources; \
       fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    tar \
    gzip \
    bash \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
