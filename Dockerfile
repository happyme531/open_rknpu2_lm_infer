FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS download

ARG TARGETARCH
ARG VERSION=0.10.0

RUN if [ "$TARGETARCH" != arm64 ]; then \
        echo 'This image requires --platform linux/arm64 (RK3588).' >&2; exit 1; \
    fi \
    && apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && curl -fL --retry 3 \
        "https://github.com/happyme531/open_rknpu2_lm_infer/releases/download/v${VERSION}/open_rknpu2_lm_infer-v${VERSION}-linux-aarch64.tar.gz" \
        -o /tmp/release.tar.gz \
    && mkdir -p /opt/open_rknpu2_lm_infer \
    && tar -xzf /tmp/release.tar.gz -C /opt/open_rknpu2_lm_infer --strip-components=1

FROM debian:bookworm-slim

ARG VERSION=0.10.0
LABEL org.opencontainers.image.source="https://github.com/happyme531/open_rknpu2_lm_infer" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=download /opt/open_rknpu2_lm_infer /opt/open_rknpu2_lm_infer
WORKDIR /opt/open_rknpu2_lm_infer
ENV RKNPU_MODEL=/models XDG_CACHE_HOME=/cache
EXPOSE 8080
ENTRYPOINT ["/opt/open_rknpu2_lm_infer/bin/open_rknpu2_lm_infer"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
