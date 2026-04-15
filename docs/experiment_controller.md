# Experiment Controller — Design Document

## Overview

Centralized experiment queue for the SGNNET multi-user, multi-machine training setup.
Five slots across three machines; multiple users submit experiments from any machine
and the controller dispatches them fairly without starving any user.

Controller lives on Mac mini (always-on). Remote users submit via HTTP or the shell wrapper.
Storage is SQLite on the shared T9 volume — no daemon dependency, survives restarts.

---

## SQLite Schema

### File
`.controller/queue.db`

### Tables

```sql
CREATE TABLE queue_entries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user         TEXT NOT NULL,
    step_name    TEXT NOT NULL,
    script_path  TEXT NOT NULL,
    args         TEXT NOT NULL DEFAULT '',
    device_pref  TEXT NOT NULL DEFAULT 'any',  -- any|cuda|mps|cpu
    slot_pref    TEXT,                          -- NULL = any slot; set = hard reservation
    priority     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'queued', -- queued|running|done|failed|cancelled
    submitted_at TEXT NOT NULL,                -- ISO8601
    launched_at  TEXT,
    finished_at  TEXT,
    result_path  TEXT
);

CREATE TABLE round_robin (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    last_user    TEXT NOT NULL DEFAULT ''
);
```

`round_robin` is a single-row table. The pointer persists across restarts.

---

## Slot → Device Mapping

| Slot         | Device | Host        |
|--------------|--------|-------------|
| mini_mps     | mps    | local       |
| mini_cpu     | cpu    | local       |
| studio_mps   | mps    | mac-studio  |
| studio_cpu   | cpu    | mac-studio  |
| 5060ti_cuda  | cuda   | 5060ti      |

`device_pref=any` is compatible with every slot.

---

## HTTP API

All endpoints on `0.0.0.0:7433`.

### POST /submit
Body (JSON):
```json
{
  "user":        "dhiraj",
  "step_name":   "step403",
  "script_path": "scripts/train_step403_delta_tier1.py",
  "args":        "--epochs 20",
  "device_pref": "cuda",
  "slot_pref":   "5060ti_cuda",
  "priority":    0
}
```
`user` is required; server returns 400 without it.
Returns: `{"id": 42, "status": "queued"}`

### GET /queue
Optional query: `?user=dhiraj`
Returns JSON array of all (or user-filtered) entries, sorted by priority desc, submitted_at asc.

### GET /status
Returns JSON: running entries per slot, queue depth per user, round-robin pointer.

### POST /cancel/<id>
Transitions entry from `queued` → `cancelled`. Returns 404 if not found, 409 if not cancellable.

---

## Scheduling Algorithm

The scheduler ticks every 30 seconds in a background thread.

```
tick():
  free_slots = parse_slot_status_sh()   # shell subprocess → FREE slots

  if not free_slots:
    return

  # Phase 1 — slot reservations
  # A slot is "reserved" if ANY pending entry has slot_pref == that slot.
  # Reserved slots may only be filled by a matching entry.
  reserved_slots = {s for s in free_slots
                    if any pending entry has slot_pref == s}

  for slot in free_slots:
    if slot in reserved_slots:
      # find the highest-priority / earliest matching entry (round-robin among users)
      entry = pick_reserved_entry(slot)
      if entry:
        launch(entry, slot)
      # else: slot stays free — we are waiting for that reserved entry
      continue

    # Phase 2 — any-slot entries, round-robin across users
    users_ordered = users_starting_after(last_launched_user)
    for user in users_ordered:
      entry = user.earliest_queued_entry_compatible_with(slot.device)
      if entry:
        launch(entry, slot)
        set last_launched_user = user
        break

pick_reserved_entry(slot):
  users_ordered = users_starting_after(last_launched_user)
  for user in users_ordered:
    for entry in user.queued_entries_with_slot_pref(slot):
      return entry   # highest priority / earliest among this user
  return None
```

### Reservation semantics (key design decision)

A free slot is "reserved" if **any** `queued` entry anywhere in any user's queue has
`slot_pref` equal to that slot. This prevents the scheduler from filling the slot with
an `any`-pref entry that would delay the intended specific experiment.

If a reserved slot has no matching entry ready (e.g. the entry is `running` elsewhere —
which can't happen since slot_pref pins to exactly one slot — or the user cancelled it),
the slot stays free until the reservation clears.

---

## Failure Modes and Recovery

| Failure | Behaviour |
|---------|-----------|
| Controller crash | SQLite persists all state. Restart picks up where it left off. Running entries that are actually still in tmux remain `running`; ones that died are reconciled on next tick via slot_status.sh. |
| `launch_slot.sh` exit 2 (slot occupied) | Controller marks entry back to `queued`, logs warning. Can happen if an out-of-band tmux session occupies the slot. |
| SSH unreachable (exit 3) | Entry stays `queued`. Logged. Slot treated as non-free on next tick. |
| Entry script crashes (non-zero exit after launch) | tmux session stays alive until DONE/FREE state detected. Controller transitions to `done` (result reporting is best-effort). |
| Multiple controller instances | Both write to same SQLite file; row-level WAL protects. However launching both is a config mistake — use the launchd plist. |
| DB corruption | Delete `.controller/queue.db`. All history is lost but in-flight tmux jobs continue. Re-queue manually. |

---

## Recovery Procedure

1. `ps aux | grep experiment_controller` — kill stale instance.
2. `python3 scripts/experiment_controller.py` — restart.
3. Controller rescans slot_status.sh on first tick. Entries in DB with `status=running`
   but whose tmux session is FREE/DONE are transitioned to `done`.
4. Entries that are genuinely still running are left as `running`.

---

## Deployment

See `scripts/experiment_controller.service.template` for the launchd plist.

Logs: `logs/experiment_controller.log`

Port: 7433 (chosen because it was free; change via `--port` argument).
