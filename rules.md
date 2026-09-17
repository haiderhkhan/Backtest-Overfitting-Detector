# Rules of Engagement

This file documents how this project gets *built* — not the statistics (see
`ARCHITECTURE.md` for that), the working process. Applies in this chat and
in Claude Code equally.

## 1. Who this is for

Haider is new to finance, not new to code — a first-year BS Accounting &
Finance student with a strong applied AI/data-science background, on his
first "hardcore" finance project. Explanations should assume CS/stats
fluency and zero finance-jargon fluency: explain finance concepts the way
you'd explain them to a smart new finance student, not skip past them.

## 2. The point of this project

The goal isn't just a working module — it's understanding *why* each
statistical method works. Every formula gets implemented from its original
paper, not copy-pasted from a library, so there's something to actually
learn from the build.

## 3. Nothing gets built without sign-off

No file — code or otherwise — gets created until Haider explicitly says go.
Design it, get agreement, then build it. Every time.

## 4. Docs before code

Order of operations: MD design docs → README → architecture doc →
PRD / rules / phases → *then* code. That order is now complete as of the
design phase — see `phases.md` for what comes next.

## 5. Ask, don't assume

When a decision affects the design (data types, thresholds, schema fields),
ask in chat as a small set of choices rather than picking silently. Keeps
the project on the same page instead of a decision surfacing after the fact.

## 6. Where things get built

- **This chat** — design, architecture, planning, decisions.
- **Claude Code in VS Code** — actual implementation, once a phase is
  signed off.

## 7. Continuity between sessions

After every Claude Code response, paste a short summary of what happened
into `memory.md`. This keeps context cheap to reload in a future session
instead of re-explaining the whole project from scratch.

## 8. Data & environment

This project starts from a clean slate — new database, no reused tables or
fixtures from any other project. PSX market data (for the factor-model
side) comes from the [psxdata API](https://psxdata.mintlify.app).

## 9. Current timeline

Design phase wrapped up September 18, 2026. Build work pauses for Haider's
mid-term exams and resumes after — see `phases.md` for what's next.
