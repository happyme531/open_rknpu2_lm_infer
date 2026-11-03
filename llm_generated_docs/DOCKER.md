# Docker deployment

Use Docker Engine and the Compose plugin on an ARM64 RK3588 Linux host with the vendor `rknpu2` kernel driver, version 0.9.6 or newer. The container uses the host kernel and NPU; it cannot provide the driver or run RK3588 inference on an x86 machine.

The prebuilt image is `ghcr.io/happyme531/open_rknpu2_lm_infer:0.11.0`. It contains the published ARM64 release in a `scratch` image with its dynamic loader, shared libraries, documentation, and license notices. The final image has no shell or package manager. No proprietary `librknnrt.so` is required.

## Start the API server

Download a supported model on the host as described in the [Quick Start](../README.md#quick-start), then run from the repository directory:

```sh
export RKNPU_MODEL_DIR=/absolute/path/to/Qwen3-4B-W4A8-Mixed-QQQ-Rot
export RKNPU_DEVICE=/dev/dri/renderD129
docker compose up -d
docker compose logs -f llm
```

`RKNPU_DEVICE` selects the host's **NPU DRM render node**, defaulting to `/dev/dri/renderD129`. Compose maps it to `/dev/dri/renderD129` inside the container. The v0.11.0 runtime discovers render nodes and verifies the `rknpu` driver identity. No `privileged` mode is needed. Check the host's device/driver association if numbering differs, and set `RKNPU_DEVICE` to the actual NPU render node.

The model directory is mounted read-only at `/models`; missing host directories cause an error. Tuning files persist in the `tune-cache` named volume at `/cache/open_rknpu2_lm_infer` across container rebuilds and recreation. `docker compose down` retains that volume; `docker compose down -v` removes it.

The Docker daemon must be able to read the host model directory. If an SSHFS mount only permits access by its mounting user, copy the model to a local directory before using it as a container mount.

The server listens on `0.0.0.0:8080` inside the container, and Compose publishes it at `http://127.0.0.1:8080` on the host by default:

```sh
curl --noproxy 127.0.0.1 http://127.0.0.1:8080/v1/models
```

To allow access from other machines, set `RKNPU_BIND_ADDRESS=0.0.0.0`. Set `OPEN_RKNPU_API_KEY` before starting Compose to require a bearer API key. `RKNPU_PORT` changes the published host port. Edit `command` in [compose.yaml](../compose.yaml) to add options such as `--ctx-size` or speculative decoding; additional model, image or expert-store paths require their own mounts.

CPU/NPU frequency settings belong to the host. Configure them there as described in the README before serving or tuning. The template uses Docker's default root user for device access; it does not mount host sysfs for frequency changes.

## Other commands and versions

Stop the service and other NPU jobs before tuning. The named volume is shared by Compose commands:

```sh
docker compose stop llm
docker compose run --rm --no-deps llm tune
docker compose up -d

# Interactive chat instead of the server
docker compose stop llm
docker compose run --rm --no-deps llm chat
```

Compose downloads the prebuilt image from GHCR; no local build is needed. Choose an image version by setting `RKNPU_VERSION` without the `v` prefix. The `latest` tag tracks the latest stable GitHub Release. Use `docker compose pull` followed by `docker compose up -d` to update.

For a local image build, the [Dockerfile](../Dockerfile) downloads the selected GitHub Release rather than compiling the checkout. Debian 12 is used only in intermediate build stages to obtain runtime libraries. A published release must provide the standard `open_rknpu2_lm_infer-vVERSION-linux-aarch64.tar.gz` asset. Build and check it without model weights or an NPU:

```sh
docker build --platform linux/arm64 --build-arg VERSION=0.11.0 \
  -t ghcr.io/happyme531/open_rknpu2_lm_infer:0.11.0 .
docker run --rm --platform linux/arm64 ghcr.io/happyme531/open_rknpu2_lm_infer:0.11.0 --version
docker run --rm --platform linux/arm64 ghcr.io/happyme531/open_rknpu2_lm_infer:0.11.0 models
```

An x86 build host needs ARM64 emulation for the runtime package installation and CLI checks. These checks establish image construction and executable startup; model loading, NPU ioctl permissions and inference still require an RK3588 host.
