# Sandbox worker image for Go tool-use evaluation.
#
# The pool runs `golang:1.23-bookworm` with a read-only root and no network. That
# base image has Go but not rtk, and five of the nine catalog tools dispatch
# through rtk, so a worker built from the stock image fails every go_build and
# go_test call with "rtk is not installed" -- a harness error attributed to the
# model. Hence this image.
#
# rtk is pinned and verified against the publisher's checksum. A worker whose
# toolchain is not provably the intended one produces scores nobody can trust.
#
# Build:  docker build -t gotooltrain/go-sandbox:0.1.0 -f docker/go-sandbox/Dockerfile .
# Verify: docker run --rm --network none gotooltrain/go-sandbox:0.1.0 selfcheck

ARG GO_IMAGE=golang:1.23-bookworm
ARG RTK_VERSION=0.50.0

FROM ${GO_IMAGE}

# The musl build is static, so it needs no loader that a read-only distroless-ish
# base would lack. The expected sha256 is pinned here *and* looked up in the
# release's own checksums.txt at build time: the pin states what was reviewed, the
# fetch proves the publisher still says so. A mismatch fails the build.
ARG RTK_VERSION
ENV RTK_SHA256=bc2b8902b0d9c796c82ef45f16ae2307e17757afeca5ee156235a3dc7bda5f89
ENV RTK_BASE_URL=https://github.com/rtk-ai/rtk/releases/download

RUN set -eux; \
    base="${RTK_BASE_URL}/v${RTK_VERSION}"; \
    asset="rtk-x86_64-unknown-linux-musl.tar.gz"; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL --retry 5 --retry-delay 3 -o /tmp/rtk.tar.gz "${base}/${asset}"; \
    curl -fsSL --retry 5 --retry-delay 3 -o /tmp/checksums.txt "${base}/checksums.txt"; \
    published="$(awk -v a="${asset}" '$2 == a { print $1 }' /tmp/checksums.txt)"; \
    test -n "${published}" || { echo "no checksum published for ${asset}" >&2; exit 1; }; \
    test "${published}" = "${RTK_SHA256}" || { \
      echo "checksum for ${asset} is ${published}, pinned ${RTK_SHA256}" >&2; exit 1; }; \
    echo "${RTK_SHA256}  /tmp/rtk.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/rtk.tar.gz -C /tmp; \
    install -m 0755 /tmp/rtk /usr/local/bin/rtk; \
    rm -f /tmp/rtk /tmp/rtk.tar.gz /tmp/checksums.txt; \
    rtk --version

# The pool mounts /tmp, the build cache and the module cache as tmpfs, so those
# are the only writable paths a worker has. Declaring them as volumes as well
# would shadow the tmpfs mounts; nothing to add here.
ENV GOFLAGS=-mod=mod
ENV GOTOOLCHAIN=local

# Fail the build, not a whole eval run, if the image is missing what it needs.
COPY docker/go-sandbox/selfcheck.sh /usr/local/bin/selfcheck
RUN chmod 0755 /usr/local/bin/selfcheck && selfcheck

CMD ["sleep", "infinity"]
