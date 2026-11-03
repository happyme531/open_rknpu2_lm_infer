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

FROM debian:bookworm-slim AS libraries

RUN apt-get update \
    && apt-get install --no-install-recommends -y libstdc++6 libgomp1

COPY --from=download /opt/open_rknpu2_lm_infer /opt/open_rknpu2_lm_infer
RUN set -eu; \
    mkdir -p /rootfs/tmp; \
    chmod 1777 /rootfs/tmp; \
    libraries="$(ldd /opt/open_rknpu2_lm_infer/bin/open_rknpu2_lm_infer)"; \
    for library in $(printf '%s\n' "$libraries" | awk '$2 == "=>" { print $3 } $1 ~ "^/" { print $1 }'); do \
        cp -L --parents "$library" /rootfs; \
    done; \
    for package in libc6 libstdc++6 libgomp1 libgcc-s1; do \
        cp -L --parents "/usr/share/doc/$package/copyright" /rootfs; \
    done; \
    cp -r --parents /usr/share/common-licenses /rootfs

FROM scratch

ARG VERSION=0.10.0
LABEL org.opencontainers.image.source="https://github.com/happyme531/open_rknpu2_lm_infer" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}"

COPY --from=libraries /rootfs /
COPY --from=download /opt/open_rknpu2_lm_infer /opt/open_rknpu2_lm_infer
WORKDIR /opt/open_rknpu2_lm_infer
ENV RKNPU_MODEL=/models XDG_CACHE_HOME=/cache
EXPOSE 8080
ENTRYPOINT ["/opt/open_rknpu2_lm_infer/bin/open_rknpu2_lm_infer"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
