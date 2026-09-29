#!/bin/sh
# Prove the image has everything a pool worker needs, before a run needs it.
#
# Run at build time so a broken image is never pushed, and available at run time
# as `selfcheck` so a worker can be interrogated when a run misbehaves. Each
# check names what it rules out, because "it worked on my machine" is the failure
# mode this replaces.
set -eu

fail() { printf 'selfcheck FAILED: %s\n' "$1" >&2; exit 1; }

# rtk: five of the nine catalog tools dispatch through it.
command -v rtk >/dev/null 2>&1 || fail "rtk is absent; go_build/go_test/grep/read_file cannot run"
rtk --version >/dev/null 2>&1 || fail "rtk is on PATH but does not run"

# go: go_build, go_test and go_doc all need it.
command -v go >/dev/null 2>&1 || fail "go is absent"
go version >/dev/null 2>&1 || fail "go is on PATH but does not run"

# The module and build caches must be writable paths, because the pool mounts
# them as tmpfs over a read-only root. If the image relocated either, the mount
# would land somewhere the toolchain does not look.
cache="$(go env GOCACHE)"
mod="$(go env GOMODCACHE)"
[ -n "$cache" ] || fail "GOCACHE is empty; the tmpfs mount would miss it"
[ -n "$mod" ] || fail "GOMODCACHE is empty; the tmpfs mount would miss it"

# An offline worker cannot fetch modules, so a task must ship vendored
# dependencies. Verify the toolchain can at least build with no network by
# compiling a module that needs nothing external.
tmp="$(mktemp -d)"
cat >"$tmp/go.mod" <<'EOM'
module example.com/selfcheck

go 1.23
EOM
cat >"$tmp/selfcheck.go" <<'EOM'
package selfcheck

func Add(a, b int) int { return a + b }
EOM
(cd "$tmp" && GOFLAGS=-mod=mod go build ./...) || fail "the image cannot build a trivial module"
rm -rf "$tmp"

printf 'selfcheck ok: rtk %s, %s, GOCACHE=%s, GOMODCACHE=%s\n' \
  "$(rtk --version)" "$(go env GOVERSION)" "$cache" "$mod"
