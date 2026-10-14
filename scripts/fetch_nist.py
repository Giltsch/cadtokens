#!/usr/bin/env python3
"""Download the NIST MBE PMI test models used by tests/test_nist.py.

The NIST CTC/FTC/STC models are the reference test set for semantic PMI in
STEP AP242. They are public domain, but too large to keep in the repository.

    python scripts/fetch_nist.py            # → tests/data/nist/*.stp
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.nist.gov/document/nist-pmi-step-files"
SHA256 = "8fa78429e6d8d9b0d7681d223b6aa9ec98c3772185c55b1a0e3679b21c181911"
DEST = Path(__file__).resolve().parent.parent / "tests" / "data" / "nist"


def main() -> int:
    print(f"Downloading {URL} ...")
    # nist.gov rejects urllib's default user agent
    request = urllib.request.Request(URL, headers={"User-Agent": "cadtokens-test-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        data = resp.read()

    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"Checksum mismatch: got {digest}, expected {SHA256}.\n"
              f"NIST may have updated the archive — check before updating SHA256.")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if "ap242" in base and base.endswith(".stp"):
                (DEST / base).write_bytes(zf.read(name))
                count += 1
    print(f"Extracted {count} AP242 files to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
