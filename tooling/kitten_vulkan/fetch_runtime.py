#!/usr/bin/env python3
"""Fetch the pinned ExecuTorch source and its minimal native test dependencies."""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

REVISION = "3b60683923245cf472b7323426920e15623ba361"  # v1.5.1
DEPENDENCIES = {
    "backends/vulkan/third-party/Vulkan-Headers": "KhronosGroup/Vulkan-Headers",
    "backends/vulkan/third-party/VulkanMemoryAllocator": "GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator",
    "backends/vulkan/third-party/volk": "zeux/volk",
    "backends/xnnpack/third-party/cpuinfo": "pytorch/cpuinfo",
    "backends/xnnpack/third-party/pthreadpool": "Maratyszcza/pthreadpool",
    "backends/xnnpack/third-party/FXdiv": "Maratyszcza/FXdiv",
    "third-party/flatbuffers": "google/flatbuffers",
    "third-party/flatcc": "dvidelabs/flatcc",
    "third-party/gflags": "gflags/gflags",
    "third-party/pocketfft": "mreineck/pocketfft",
    "third-party/json": "nlohmann/json",
}


def archive(repo, revision, dest):
    marker = dest / ".source-revision"
    if marker.exists() and marker.read_text().strip() == revision:
        return
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    data = urllib.request.urlopen(url, timeout=120).read()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        prefix = members[0].name.split("/")[0] + "/"
        for member in members:
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            tar.extract(member, dest, filter="data")
    marker.write_text(revision + "\n")
    print(repo, revision, hashlib.sha256(data).hexdigest(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive("pytorch/executorch", REVISION, args.output)
    url = f"https://api.github.com/repos/pytorch/executorch/git/trees/{REVISION}?recursive=1"
    tree = json.load(urllib.request.urlopen(url, timeout=60))["tree"]
    revisions = {item["path"]: item["sha"] for item in tree if item["type"] == "commit"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(archive, repo, revisions[path], args.output / path)
                 for path, repo in DEPENDENCIES.items()]
        for task in tasks:
            task.result()
    (args.output / "source-lock.json").write_text(json.dumps(
        {"executorch": REVISION, "dependencies": {p: revisions[p] for p in DEPENDENCIES}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
