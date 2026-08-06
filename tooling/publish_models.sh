#!/usr/bin/env bash
# Publish converted GGUFs as GitHub release assets so the app can download
# them (upstream publishes no VoiceDesign GGUF). Release assets are capped at
# 2 GB each, which our ~1 GB talker and ~0.4 GB codec fit inside.
#
#   GH_TOKEN=ghp_... tooling/publish_models.sh Qwen3-TTS-VD-Q4_K_M.gguf mmproj-Qwen3-TTS-VD-Q8_0.gguf
#
# The tag must match MODEL_RELEASE in ModelManager.kt.
set -euo pipefail

REPO="${REPO:-maxfridbe/vibe_android_tts_runner}"
TAG="${TAG:-models-v1}"
: "${GH_TOKEN:?set GH_TOKEN to a token with 'contents: write' on $REPO}"
[ "$#" -gt 0 ] || { echo "usage: $0 <file.gguf> [more.gguf...]" >&2; exit 1; }

api() { curl -sS -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$@"; }

echo "==> ensuring release $TAG exists"
id=$(api "https://api.github.com/repos/$REPO/releases/tags/$TAG" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
if [ -z "$id" ]; then
    id=$(api -X POST "https://api.github.com/repos/$REPO/releases" \
        -d "{\"tag_name\":\"$TAG\",\"name\":\"Model weights\",\"body\":\"GGUF conversions the app downloads at runtime.\",\"draft\":false,\"prerelease\":false}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
    echo "    created release id=$id"
else
    echo "    reusing release id=$id"
fi

for f in "$@"; do
    name=$(basename "$f")
    echo "==> uploading $name ($(du -h "$f" | cut -f1))"
    # replace an existing asset of the same name
    old=$(api "https://api.github.com/repos/$REPO/releases/$id/assets" \
        | python3 -c "import json,sys;print(next((a['id'] for a in json.load(sys.stdin) if a['name']=='$name'),''))")
    [ -n "$old" ] && api -X DELETE "https://api.github.com/repos/$REPO/releases/assets/$old" >/dev/null
    curl -sS --fail -H "Authorization: Bearer $GH_TOKEN" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$f" \
        "https://uploads.github.com/repos/$REPO/releases/$id/assets?name=$name" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print("    ok:", d["browser_download_url"])'
done
