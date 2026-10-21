# Architecture decision records

Decisions that shape how purecv is built or maintained, one file each, in short
[MADR](https://adr.github.io/madr/) form: Status, Date, Deciders, Context, Decision,
Alternatives considered, Consequences, Links.

## Convention

- Files are numbered in order: `NNNN-short-title.md` (`0001`, `0002`, …). Numbers are never
  reused.
- An accepted ADR is never edited, except its **Status** line.
- A decision that changes is a **new** ADR that says which one it supersedes; the old one's
  Status becomes `superseded by NNNN`.
- Status values: `proposed`, `accepted`, `deprecated`, `superseded by NNNN`.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-opencv-parity-audit.md) | OpenCV parity audit with one pinned reference version and frozen per-version records | accepted |
