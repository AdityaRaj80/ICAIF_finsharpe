"""PREREG binding-freeze stamp (PREREGISTRATION.md §12 k.1).

At the FIRST Phase-2 step (HPO or training) the protocol becomes
immutable. `ensure_frozen()`:
  (1) verifies PREREGISTRATION.md still hashes to the SHA recorded in
      PREREGISTRATION.sha256 (refuses to run if the doc was edited);
  (2) writes an append-only FREEZE_STAMP (utc, sha, host, job) exactly
      once, race-safe via O_EXCL across concurrent SLURM array tasks;
  (3) on every later call re-verifies the live doc still matches the
      stamped SHA -> aborts on any post-freeze edit.

k.1 is honor-bound and DISCLOSED, not externally enforceable (no `.git`
authority on the cluster; author-controlled SHA). This module makes the
honour mechanical and tamper-evident, nothing more.
"""
import datetime
import hashlib
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Real campaign uses the repo files. The local smoke test points these
# at a throwaway tmp dir (P2_FREEZE_*) so it can NOT prematurely lock the
# real protocol — the real freeze fires only when the real paths are used.
DOC = os.environ.get("P2_FREEZE_DOC", os.path.join(ROOT,
                                                   "PREREGISTRATION.md"))
SHAFILE = os.environ.get("P2_FREEZE_SHA",
                         os.path.join(ROOT, "PREREGISTRATION.sha256"))
STAMP = os.environ.get("P2_FREEZE_STAMP",
                       os.path.join(ROOT, "FREEZE_STAMP"))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_sha():
    with open(SHAFILE) as f:
        return f.read().split()[0].strip()


def ensure_frozen(job="phase2"):
    """Idempotent. Returns the frozen SHA. Aborts the process on any
    integrity violation (this is intentional: a broken freeze must stop
    the campaign, not warn)."""
    live = sha256_of(DOC)
    rec = recorded_sha()
    if live != rec:
        sys.exit(f"FREEZE-ABORT: PREREGISTRATION.md sha={live} != "
                 f"recorded={rec}. Protocol doc edited; integrity broken "
                 f"(k.1). Refusing to train.")
    if os.path.exists(STAMP):
        with open(STAMP) as f:
            head = f.readline()
        if rec not in head:
            sys.exit("FREEZE-ABORT: FREEZE_STAMP SHA mismatch -> "
                     "post-freeze edit detected (k.1).")
        return rec
    line = (f"{rec}  frozen_utc="
            f"{datetime.datetime.utcnow().isoformat()}Z  "
            f"host={socket.gethostname()}  job={job}\n")
    try:                                   # first array task wins the race
        fd = os.open(STAMP, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        os.write(fd, line.encode())
        os.close(fd)
    except FileExistsError:                # another task already stamped
        with open(STAMP) as f:
            if rec not in f.readline():
                sys.exit("FREEZE-ABORT: concurrent stamp SHA mismatch.")
    return rec


if __name__ == "__main__":
    s = ensure_frozen("selftest")
    print(f"[freeze] doc sha verified = {s}")
    print(f"[freeze] stamp: {open(STAMP).read().strip()}")
    print("FREEZE-SELFTEST: PASS")
