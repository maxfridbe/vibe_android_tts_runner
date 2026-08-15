#!/usr/bin/env bash
# Build one or more apps from apps/ inside the shared Docker/Podman image.
#
#   ./build.sh              build every app under apps/
#   ./build.sh hello-kotlin build a single app
#   ./build.sh a b c        build a selection
#
# Environment overrides:
#   BUILD_TASK=assembleRelease   gradle task(s) to run   (default: assembleDebug)
#   IMAGE=androidbase-builder    builder image tag
#   GRADLE_CACHE_VOLUME=...      named volume for the gradle cache
#   SKIP_IMAGE_BUILD=1           assume the image already exists (used by CI)
#   CONTAINER_ENGINE=docker      force an engine instead of autodetecting
#   VERSION_CODE / VERSION_NAME  override the git-derived app version
#   KEYSTORE_DIR=...             where release.keystore + keystore.properties
#                                live (default: ./keystore, committed so every
#                                clone and CI signs with the same key)
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-androidbase-builder}"
GRADLE_CACHE_VOLUME="${GRADLE_CACHE_VOLUME:-androidbase-gradle-cache}"
BUILD_TASK="${BUILD_TASK:-assembleDebug}"

# Automated versioning from the build clock (UTC): yy.mmdd.hhmm — reads as a
# date, sorts as a number, and only ever goes up, so a reinstall from any
# machine can never hit INSTALL_FAILED_VERSION_DOWNGRADE (the old commit-count
# scheme did whenever checkouts disagreed on history). versionCode packs the
# same stamp into an int31: (yy-20)*1e8 + mmddhhmm — good until 2041 (the
# 2.1e9 Android cap).
STAMP_UTC="$(date -u +%y%m%d%H%M)"
VERSION_CODE="${VERSION_CODE:-$(( (10#${STAMP_UTC:0:2} - 20) * 100000000 + 10#${STAMP_UTC:2} ))}"
VERSION_NAME="${VERSION_NAME:-$(date -u +%y.%m%d.%H%M)}"
VERSION_NAME="${VERSION_NAME#v}"
echo "==> Version: name=$VERSION_NAME code=$VERSION_CODE"

# CI runners can have BOTH engines installed while only one holds the
# builder image, so CONTAINER_ENGINE overrides autodetection.
ENGINE="${CONTAINER_ENGINE:-}"
if [ -z "$ENGINE" ]; then
    if command -v podman >/dev/null 2>&1; then
        ENGINE=podman
    elif command -v docker >/dev/null 2>&1; then
        ENGINE=docker
    else
        echo "error: neither podman nor docker found on PATH" >&2
        exit 1
    fi
fi
case "$ENGINE" in
    # SELinux relabel flag for bind mounts (Fedora et al.)
    podman) ZFLAG=":Z" ;;
    *)      ZFLAG="" ;;
esac

# Select apps: arguments, or every directory under apps/.
if [ "$#" -gt 0 ]; then
    APPS=("$@")
else
    APPS=()
    for dir in apps/*/; do
        APPS+=("$(basename "$dir")")
    done
fi

if [ "${SKIP_IMAGE_BUILD:-0}" != "1" ]; then
    echo "==> Building builder image: $IMAGE"
    "$ENGINE" build -t "$IMAGE" docker
fi

# --- Signing ---------------------------------------------------------------
# One self-signed keystore shared by every app so debug and release builds
# always carry the same signature (reinstalls just work). Generated once via
# keytool inside the builder image and stored in $KEYSTORE_DIR.
KEYSTORE_DIR="${KEYSTORE_DIR:-$(pwd)/keystore}"
KEYSTORE_FILE="$KEYSTORE_DIR/release.keystore"
KEYSTORE_PROPS="$KEYSTORE_DIR/keystore.properties"

if [ ! -f "$KEYSTORE_FILE" ] || [ ! -f "$KEYSTORE_PROPS" ]; then
    echo "==> No keystore found; generating one in $KEYSTORE_DIR"
    mkdir -p "$KEYSTORE_DIR"
    # Finite read (a bare `tr </dev/urandom | head` dies of SIGPIPE under pipefail)
    pass="$(head -c 48 /dev/urandom | base64 | tr -d '/+=\n' | head -c 32)"
    printf 'storePassword=%s\nkeyPassword=%s\nkeyAlias=androidbase\n' \
        "$pass" "$pass" > "$KEYSTORE_PROPS"
    "$ENGINE" run --rm -v "$KEYSTORE_DIR:/ks$ZFLAG" "$IMAGE" \
        keytool -genkeypair -keystore /ks/release.keystore -alias androidbase \
        -keyalg RSA -keysize 4096 -validity 10950 \
        -storepass "$pass" -keypass "$pass" \
        -dname "CN=AndroidBase"
fi

storePassword="" keyPassword="" keyAlias=""
. "$KEYSTORE_PROPS"

for app in "${APPS[@]}"; do
    dir="apps/$app"
    if [ ! -d "$dir" ]; then
        echo "error: no such app: $dir" >&2
        exit 1
    fi

    echo "==> Building $app ($BUILD_TASK)"
    # ORG_GRADLE_PROJECT_* env vars become Gradle project properties.
    "$ENGINE" run --rm \
        -v "$(pwd)/$dir:/workspace$ZFLAG" \
        -v "$GRADLE_CACHE_VOLUME:/home/gradle/.gradle" \
        -v "$KEYSTORE_FILE:/keystore.jks$ZFLAG" \
        -e "ORG_GRADLE_PROJECT_appVersionCode=$VERSION_CODE" \
        -e "ORG_GRADLE_PROJECT_appVersionName=$VERSION_NAME" \
        -e "ORG_GRADLE_PROJECT_appKeystoreFile=/keystore.jks" \
        -e "ORG_GRADLE_PROJECT_appKeystorePassword=$storePassword" \
        -e "ORG_GRADLE_PROJECT_appKeyAlias=$keyAlias" \
        -e "ORG_GRADLE_PROJECT_appKeyPassword=$keyPassword" \
        -w /workspace \
        "$IMAGE" gradle --no-daemon $BUILD_TASK

    mkdir -p "output/$app"
    found=0
    # APKs (assemble*) and AABs (bundle*, the Play upload format) both land here
    while IFS= read -r art; do
        found=1
        # app-debug.apk -> <app>-<version>-debug.apk ; app-release.aab -> <app>-<version>-release.aab
        cp -v "$art" "output/$app/${app}-${VERSION_NAME}-$(basename "$art" | sed 's/^app-//')"
    done < <(find "$dir" \( -path '*/build/outputs/apk/*' -name '*.apk' \) -o \
                        \( -path '*/build/outputs/bundle/*' -name '*.aab' \))
    if [ "$found" = "0" ]; then
        echo "error: no APK or AAB produced for $app" >&2
        exit 1
    fi
done

echo "==> Done. APKs in output/:"
find output -name '*.apk'
