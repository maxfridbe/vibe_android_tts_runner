#!/usr/bin/env python3
"""Download the app's pinned Kitten decoder/voices, verifying exact contents."""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

REVISION = "d820e8476c35e637dc5c89a1e66c345f620bd0c0"
ASSETS = {
    "decoder.pt": "47ea803ad46a87f876c1b3b02f05e93dad15e4d2b58eafeb34884f52cdec0268",
    "voices.json": "947d3f85da00e6b9f591c456cf5dd14efcb16b2b5cf3418bc6ecc8a540a027b4",
}


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        destination = args.output / name
        if destination.exists() and checksum(destination) == expected:
            print(f"Verified {name}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        url = f"https://huggingface.co/KittenML/kitten-tts-2/resolve/{REVISION}/cpp/student_w4/{name}"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if checksum(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    main()
