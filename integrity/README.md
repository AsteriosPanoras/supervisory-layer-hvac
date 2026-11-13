# Integrity

- **`ledger_sha256.txt`:** the append-only ledger of the study.
  - Every protocol, analysis script and result file was hashed (sha256) before the outcome it concerns, with the time of hashing.
  - The comment lines were cleaned for publication; the hash lines are unchanged.
- **`opentimestamps/`:** OpenTimestamps proofs for the later registrations (`*.ots`) and the lists of hashes they stamp (`SHA256SUMS*.txt`).
  - A proof shows that the listed hashes existed at that time. Check it with `ots verify`, given the original files.
- **`EXPORT_MAP.csv`:** for every published file, its source, the sha256 of the original and of the published copy, and how many lines were changed.
  - Published copies differ from the originals only where working notes or local paths were removed.
  - The originals stay in the authors' archive, so each hash in the ledger can be checked against them.
