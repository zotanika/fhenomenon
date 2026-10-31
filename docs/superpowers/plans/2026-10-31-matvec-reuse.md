# Matvec reuse implementation plan

User authorized implementation/investigation; no additional approval gate.
Primary checkout: /Users/sunchuljung/repo/zotanika/fhenomenon.
Design: ../specs/2026-10-31-matvec-reuse-design.md.

- [x] Profile unchanged matvec phases and publication costs; retain raw evidence.
- [x] Add exact optional prepared arithmetic paths and differential tests.
- [x] Add native preparation/consumption API, owned artifact and lifecycle tests.
- [x] Add independent contribution registration and budget/identity/interop tests.
- [x] Integrate benchmark ablations and installed public consumer.
- [x] Independent Astra review, Debug/sanitizer/Release checks, frozen hashes.
- [x] Isolated timing, complete-app oracle, setup/memory/break-even report.

Result: docs/matvec-preparation-validation-2026-10-31.md. Debug and ASan/UBSan
43/43 CTests each; 410 GoogleTests across 35 binaries. Release/test-off and
installed consumer pass. All six frozen hashes match. Isolated 24-run median:
immediate 80.532 ms, old coarse 79.444 ms, prepared full 39.444 ms, mixed
prepared/ordinary split 47.999 ms; full setup 6.448 ms, artifact 6.84375 MiB.
