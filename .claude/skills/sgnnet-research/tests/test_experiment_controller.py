#!/usr/bin/env python3
"""
test_experiment_controller.py — Unit tests for experiment controller scheduler logic.

Run:
    python3 -m unittest tests/test_experiment_controller.py -v
"""

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiment_controller import (
    device_compatible,
    distinct_users,
    get_free_slots,
    init_schema,
    insert_entry,
    get_queued_entries,
    open_db,
    parse_slot_status_output,
    pick_any_entry,
    pick_reserved_entry,
    round_robin_order,
    scheduler_tick,
    set_last_user,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_row(user, step_name, device_pref="any", slot_pref=None, priority=0, entry_id=None):
    """Return a mock sqlite3.Row-like dict."""
    row = MagicMock()
    row.__getitem__ = lambda self, k: {
        "id": entry_id or 1,
        "user": user,
        "step_name": step_name,
        "device_pref": device_pref,
        "slot_pref": slot_pref,
        "priority": priority,
    }[k]
    row.keys = lambda: ["id", "user", "step_name", "device_pref", "slot_pref", "priority"]
    return row


def fresh_db():
    """Return an in-memory SQLite DB with schema."""
    conn = open_db(Path(":memory:"))
    init_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Test: parse_slot_status_output
# ---------------------------------------------------------------------------

class TestParseSlotStatusOutput(unittest.TestCase):

    def test_free_slots_extracted(self):
        output = """\
===== SGNNET Slot Status =====
mini_mps      [FREE]
mini_cpu      [RUNNING]    session=sgn-dhiraj-mini_cpu-step403
studio_mps    [FREE]
studio_cpu    [DONE]       session=sgn-alice-studio_cpu-step100  (script finished)
5060ti_cuda   [FREE]
"""
        result = parse_slot_status_output(output)
        self.assertIn("mini_mps", result)
        self.assertIn("studio_mps", result)
        self.assertIn("5060ti_cuda", result)
        self.assertNotIn("mini_cpu", result)
        self.assertNotIn("studio_cpu", result)

    def test_all_running_returns_empty(self):
        output = """\
mini_mps      [RUNNING]    session=sgn-x-mini_mps-y
mini_cpu      [RUNNING]    session=sgn-x-mini_cpu-y
studio_mps    [RUNNING]    session=sgn-x-studio_mps-y
studio_cpu    [RUNNING]    session=sgn-x-studio_cpu-y
5060ti_cuda   [RUNNING]    session=sgn-x-5060ti_cuda-y
"""
        self.assertEqual(parse_slot_status_output(output), [])

    def test_all_free(self):
        output = """\
mini_mps      [FREE]
mini_cpu      [FREE]
studio_mps    [FREE]
studio_cpu    [FREE]
5060ti_cuda   [FREE]
"""
        result = parse_slot_status_output(output)
        self.assertEqual(len(result), 5)


# ---------------------------------------------------------------------------
# Test: device_compatible
# ---------------------------------------------------------------------------

class TestDeviceCompatible(unittest.TestCase):

    def test_any_is_always_compatible(self):
        for slot in ["mini_mps", "mini_cpu", "studio_mps", "studio_cpu", "5060ti_cuda"]:
            self.assertTrue(device_compatible("any", slot))

    def test_cuda_only_matches_5060ti(self):
        self.assertTrue(device_compatible("cuda", "5060ti_cuda"))
        self.assertFalse(device_compatible("cuda", "mini_mps"))
        self.assertFalse(device_compatible("cuda", "mini_cpu"))

    def test_mps_matches_mps_slots(self):
        self.assertTrue(device_compatible("mps", "mini_mps"))
        self.assertTrue(device_compatible("mps", "studio_mps"))
        self.assertFalse(device_compatible("mps", "mini_cpu"))
        self.assertFalse(device_compatible("mps", "5060ti_cuda"))

    def test_cpu_matches_cpu_slots(self):
        self.assertTrue(device_compatible("cpu", "mini_cpu"))
        self.assertTrue(device_compatible("cpu", "studio_cpu"))
        self.assertFalse(device_compatible("cpu", "5060ti_cuda"))


# ---------------------------------------------------------------------------
# Test: round_robin_order
# ---------------------------------------------------------------------------

class TestRoundRobinOrder(unittest.TestCase):

    def test_starts_after_last_user(self):
        users = ["alice", "bob", "charlie"]
        order = round_robin_order(users, "alice")
        self.assertEqual(order[0], "bob")

    def test_wraps_around(self):
        users = ["alice", "bob", "charlie"]
        order = round_robin_order(users, "charlie")
        self.assertEqual(order[0], "alice")

    def test_unknown_last_user_starts_from_zero(self):
        users = ["alice", "bob"]
        order = round_robin_order(users, "unknown")
        self.assertEqual(order, ["alice", "bob"])

    def test_single_user(self):
        order = round_robin_order(["alice"], "alice")
        self.assertEqual(order, ["alice"])

    def test_empty_users(self):
        self.assertEqual(round_robin_order([], "alice"), [])


# ---------------------------------------------------------------------------
# Test: pick_reserved_entry
# ---------------------------------------------------------------------------

class TestPickReservedEntry(unittest.TestCase):

    def _make_rows(self, specs):
        """specs: list of (user, step, device_pref, slot_pref, priority, id)"""
        rows = []
        for i, (user, step, dp, sp, prio) in enumerate(specs):
            row = make_row(user, step, dp, sp, prio, entry_id=i + 1)
            rows.append(row)
        return rows

    def test_no_reservation_returns_none(self):
        rows = self._make_rows([
            ("alice", "s1", "any", None, 0),
            ("bob",   "s2", "any", None, 0),
        ])
        result = pick_reserved_entry(rows, "5060ti_cuda", "")
        self.assertIsNone(result)

    def test_picks_matching_slot_entry(self):
        rows = self._make_rows([
            ("alice", "s1", "cuda", "5060ti_cuda", 0),
            ("bob",   "s2", "any",  None,          0),
        ])
        result = pick_reserved_entry(rows, "5060ti_cuda", "")
        self.assertIsNotNone(result)
        self.assertEqual(result["user"], "alice")

    def test_round_robin_among_reserved_entries(self):
        rows = self._make_rows([
            ("alice", "sA", "mps", "mini_mps", 0),
            ("bob",   "sB", "mps", "mini_mps", 0),
        ])
        # last_user=alice → bob should go first
        result = pick_reserved_entry(rows, "mini_mps", "alice")
        self.assertEqual(result["user"], "bob")

    def test_higher_priority_wins_within_user(self):
        rows = self._make_rows([
            ("alice", "s_low",  "cuda", "5060ti_cuda", 0),
            ("alice", "s_high", "cuda", "5060ti_cuda", 10),
        ])
        # Both Alice's, but s_low comes first in list (submitted earlier).
        # In a real DB, sorted priority DESC so s_high would be first.
        # Here we mimic by putting high-priority first.
        rows_sorted = [rows[1], rows[0]]  # high first
        result = pick_reserved_entry(rows_sorted, "5060ti_cuda", "")
        self.assertEqual(result["step_name"], "s_high")


# ---------------------------------------------------------------------------
# Test: pick_any_entry with round-robin fairness
# ---------------------------------------------------------------------------

class TestPickAnyEntry(unittest.TestCase):

    def _make_rows(self, specs):
        rows = []
        for i, (user, step, dp, sp, prio) in enumerate(specs):
            row = make_row(user, step, dp, sp, prio, entry_id=i + 1)
            rows.append(row)
        return rows

    def test_picks_compatible_entry(self):
        rows = self._make_rows([
            ("alice", "s1", "any", None, 0),
        ])
        result = pick_any_entry(rows, "5060ti_cuda", "")
        self.assertIsNotNone(result)
        self.assertEqual(result["user"], "alice")

    def test_skips_incompatible_device(self):
        rows = self._make_rows([
            ("alice", "s1", "mps", None, 0),  # incompatible with cuda slot
        ])
        result = pick_any_entry(rows, "5060ti_cuda", "")
        self.assertIsNone(result)

    def test_skips_slot_pref_entries(self):
        rows = self._make_rows([
            ("alice", "s1", "any", "5060ti_cuda", 0),  # has slot_pref, not any-pref
        ])
        result = pick_any_entry(rows, "5060ti_cuda", "")
        self.assertIsNone(result)

    def test_round_robin_fairness(self):
        # After alice dispatched last, bob should go next
        rows = self._make_rows([
            ("alice", "sA", "any", None, 0),
            ("bob",   "sB", "any", None, 0),
        ])
        result = pick_any_entry(rows, "mini_mps", "alice")
        self.assertEqual(result["user"], "bob")

    def test_falls_back_to_alice_if_bob_has_no_compatible(self):
        # Bob only has cuda entries, slot is mps — falls back to alice
        rows = self._make_rows([
            ("alice", "sA", "any",  None, 0),
            ("bob",   "sB", "cuda", None, 0),
        ])
        result = pick_any_entry(rows, "mini_mps", "bob")
        self.assertEqual(result["user"], "alice")


# ---------------------------------------------------------------------------
# Test: slot reservation blocks any-pref entries (integration)
# ---------------------------------------------------------------------------

class TestReservationBlocksAnyPref(unittest.TestCase):
    """
    If a free slot has a reservation (any queued entry has slot_pref == slot),
    scheduler_tick must NOT launch an any-pref entry on that slot.
    """

    def setUp(self):
        self.conn = fresh_db()
        self.db_lock = threading.Lock()

    def test_reserved_slot_not_filled_by_any_pref(self):
        # Alice has a slot-specific entry for 5060ti_cuda
        insert_entry(self.conn, {
            "user": "alice", "step_name": "alice_cuda", "script_path": "s.py",
            "args": "", "device_pref": "cuda", "slot_pref": "5060ti_cuda", "priority": 0,
        })
        # Bob has an any-pref entry
        insert_entry(self.conn, {
            "user": "bob", "step_name": "bob_any", "script_path": "s.py",
            "args": "", "device_pref": "any", "slot_pref": None, "priority": 0,
        })

        launched_entries = []

        def mock_launch(conn, entry, slot):
            launched_entries.append((dict(entry), slot))
            from scripts.experiment_controller import set_entry_status, now_iso
            set_entry_status(conn, entry["id"], "running", {"launched_at": now_iso()})
            return True

        with patch("scripts.experiment_controller.get_free_slots", return_value=["5060ti_cuda"]):
            with patch("scripts.experiment_controller.launch_entry", side_effect=mock_launch):
                scheduler_tick(self.conn, self.db_lock)

        # Only Alice's cuda entry should have been launched, not Bob's any-pref
        self.assertEqual(len(launched_entries), 1)
        self.assertEqual(launched_entries[0][0]["user"], "alice")
        self.assertEqual(launched_entries[0][1], "5060ti_cuda")

    def test_any_pref_fills_unreserved_slot(self):
        # No reservations; Bob's any-pref should be launched
        insert_entry(self.conn, {
            "user": "bob", "step_name": "bob_any", "script_path": "s.py",
            "args": "", "device_pref": "any", "slot_pref": None, "priority": 0,
        })

        launched_entries = []

        def mock_launch(conn, entry, slot):
            launched_entries.append((dict(entry), slot))
            from scripts.experiment_controller import set_entry_status, now_iso
            set_entry_status(conn, entry["id"], "running", {"launched_at": now_iso()})
            return True

        with patch("scripts.experiment_controller.get_free_slots", return_value=["mini_mps"]):
            with patch("scripts.experiment_controller.launch_entry", side_effect=mock_launch):
                scheduler_tick(self.conn, self.db_lock)

        self.assertEqual(len(launched_entries), 1)
        self.assertEqual(launched_entries[0][0]["user"], "bob")
        self.assertEqual(launched_entries[0][1], "mini_mps")


if __name__ == "__main__":
    unittest.main(verbosity=2)
