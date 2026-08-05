"""Anchors: frozen opponents that eviction cannot remove.

The bank cannot measure progress -- it evicts the entry we beat most, so its win
rate sits near 0.50 by construction -- and the scripted bots saturate. These
tests pin the two properties that make an anchor a yardstick instead: its
weights load unchanged on the policy path, and it is never evicted or reset.
"""
import argparse
import pickle
import zipfile

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from training import anchors
from training.actions import num_actions
from training.features import NUM_GLOBALS, NUM_PLANES
from training.network import NUM_OPPONENT_KINDS, ActorCritic
from training.train import (
    NETWORK,
    SCRIPTED,
    choose_opponent,
    init_league,
    league_stats,
    league_table,
    sync_anchors,
)

BOARD = 12


@pytest.fixture(scope="module")
def reference():
    net = ActorCritic(channels=4, blocks=1, groups=2)
    params = net.init(
        jrandom.PRNGKey(0),
        jnp.zeros((BOARD, BOARD, NUM_PLANES), jnp.float32),
        jnp.zeros((NUM_GLOBALS,), jnp.float32),
        jnp.ones((num_actions(BOARD, BOARD),), bool),
        jnp.zeros((NUM_OPPONENT_KINDS,), jnp.float32),
    )
    return net, jax.tree.map(np.asarray, params)


def _flat(tree, prefix=""):
    return dict(anchors._leaves(tree, prefix))


def _write_npz(path, params, kinds=NUM_OPPONENT_KINDS):
    """Export like training/export.py, optionally with a wider value head -- the
    shape ppo_v11 has, from a run with 6 opponent kinds."""
    flat = _flat(params["params"], "")
    if kinds != NUM_OPPONENT_KINDS:
        kernel = flat["Dense_2/kernel"]
        pooled = kernel.shape[0] - NUM_OPPONENT_KINDS
        wider = np.zeros((pooled + kinds, kernel.shape[1]), np.float32)
        wider[:pooled] = kernel[:pooled]
        wider[pooled:] = 1.0            # junk in the slots we must NOT copy
        flat["Dense_2/kernel"] = wider
    flat["meta/channels"] = np.asarray(4.0, np.float32)
    np.savez_compressed(path, **flat)


def test_npz_and_zip_and_pkl_all_load(tmp_path, reference):
    _, ref = reference
    npz = tmp_path / "exported.npz"
    _write_npz(npz, ref)

    bundle = tmp_path / "ppo_v11_submission.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.write(npz, "weights.npz")

    ckpt = tmp_path / "run.pkl"
    with open(ckpt, "wb") as fh:
        pickle.dump({"params": ref, "update": 10}, fh)

    loaded = anchors.load(f"{npz},{bundle},mine={ckpt}", ref)
    assert [a.name for a in loaded] == ["exported", "ppo_v11_submission", "mine"]
    for a in loaded:
        assert jax.tree.structure(a.params) == jax.tree.structure(ref)
        for got, want in zip(jax.tree.leaves(a.params), jax.tree.leaves(ref)):
            assert got.shape == want.shape


def test_wider_value_head_conforms_and_leaves_the_policy_untouched(tmp_path, reference):
    """ppo_v11's value head is 2 rows wider (6 opponent kinds, not 4). Only that
    kernel may change, and only in its one-hot rows: the actor never sees the
    opponent input, so a conformed anchor must play the identical policy."""
    net, ref = reference
    wide = tmp_path / "v11.npz"
    _write_npz(wide, ref, kinds=6)

    (anchor,) = anchors.load(str(wide), ref)
    source = _flat(anchors.read_params(str(wide)))
    got = _flat(anchor.params)

    changed = [k for k in source if source[k].shape != got[k].shape
               or not np.array_equal(source[k], got[k])]
    assert changed == ["params/Dense_2/kernel"]

    kernel = got["params/Dense_2/kernel"]
    pooled = kernel.shape[0] - NUM_OPPONENT_KINDS
    assert np.array_equal(kernel[:pooled], source["params/Dense_2/kernel"][:pooled])
    assert not kernel[pooled:].any(), "junk one-hot rows must be dropped, not copied"

    # The real requirement: it runs, and the logits match the reference weights.
    args = (jnp.zeros((BOARD, BOARD, NUM_PLANES), jnp.float32),
            jnp.zeros((NUM_GLOBALS,), jnp.float32),
            jnp.ones((num_actions(BOARD, BOARD),), bool),
            jnp.zeros((NUM_OPPONENT_KINDS,), jnp.float32))
    assert np.allclose(net.apply(anchor.params, *args)[0], net.apply(ref, *args)[0])


def test_real_architecture_change_is_refused(tmp_path, reference):
    _, ref = reference
    broken = jax.tree.map(np.asarray, ref)
    broken["params"]["Conv_0"]["kernel"] = np.zeros((3, 3, NUM_PLANES, 999), np.float32)
    path = tmp_path / "wrong.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"params": broken}, fh)
    with pytest.raises(SystemExit, match="Conv_0"):
        anchors.load(str(path), ref)


def test_duplicate_names_and_missing_files_are_refused(tmp_path, reference):
    _, ref = reference
    npz = tmp_path / "a.npz"
    _write_npz(npz, ref)
    with pytest.raises(SystemExit, match="used twice"):
        anchors.load(f"x={npz},x={npz}", ref)
    with pytest.raises(SystemExit, match="no such file"):
        anchors.load(str(tmp_path / "nope.pkl"), ref)


# --- league bookkeeping -------------------------------------------------------

class FakeAnchor:
    def __init__(self, name):
        self.name = name


def _args(**kw):
    return argparse.Namespace(
        bot_floor=0.05, anchor_floor=0.25, pfsp_power=2.0, **kw
    )


def _beaten(league):
    """Bots comfortably beaten, so choose_opponent leaves bootstrap mode."""
    for name in SCRIPTED:
        league["bots"][name] = [90.0, 0.0, 100.0]
    return league


def test_floors_split_the_roll_into_exclusive_shares():
    frozen = [FakeAnchor("v11"), FakeAnchor("cwnh")]
    bank = [object() for _ in range(10)]
    league = _beaten(init_league(bank, frozen))
    args = _args()

    picks = {"anchor": 0, "bot": 0, "bank": 0}
    for i in range(4000):
        opp = choose_opponent(args, bank, frozen, league, jrandom.PRNGKey(i))
        if opp.anchor_idx >= 0:
            picks["anchor"] += 1
            assert opp.kind == NETWORK and opp.bank_idx == -1
        elif opp.bank_idx >= 0:
            picks["bank"] += 1
            assert opp.kind == NETWORK
        else:
            picks["bot"] += 1
            assert opp.label in SCRIPTED

    total = sum(picks.values())
    assert picks["anchor"] / total == pytest.approx(0.25, abs=0.03)
    assert picks["bot"] / total == pytest.approx(0.05, abs=0.02)
    assert picks["bank"] / total == pytest.approx(0.70, abs=0.03)


def test_bots_keep_their_floor_when_they_are_all_won():
    """"Minimally, based on the win%": at 100% PFSP gives the bots the smallest
    weight it can, but --bot-floor still guarantees the sanity check runs."""
    frozen = [FakeAnchor("v11")]
    bank = [object() for _ in range(10)]
    league = init_league(bank, frozen)
    for name in SCRIPTED:
        league["bots"][name] = [100.0, 0.0, 100.0]      # every bot at 1.00
    seen = 0
    for i in range(2000):
        if choose_opponent(_args(), bank, frozen, league, jrandom.PRNGKey(i)).kind != NETWORK:
            seen += 1
    assert seen / 2000 == pytest.approx(0.05, abs=0.02)


def test_no_bank_falls_through_to_the_bots():
    frozen = [FakeAnchor("v11")]
    league = _beaten(init_league([], frozen))
    kinds = [choose_opponent(_args(), [], frozen, league, jrandom.PRNGKey(i))
             for i in range(500)]
    assert all(o.bank_idx == -1 for o in kinds)
    assert {o.kind != NETWORK for o in kinds} == {True, False}   # bots and anchors, no bank


def test_bootstrap_never_draws_an_anchor():
    """A policy that cannot beat expander learns nothing from a finished
    submission except how to lose."""
    frozen = [FakeAnchor("v11")]
    league = init_league([object()], frozen)        # nothing beaten yet
    for i in range(300):
        opp = choose_opponent(_args(), [object()], frozen, league, jrandom.PRNGKey(i))
        assert opp.anchor_idx == -1 and opp.label in SCRIPTED


def test_results_are_recorded_against_the_right_row():
    frozen = [FakeAnchor("v11"), FakeAnchor("cwnh")]
    league = init_league([object(), object()], frozen)
    from training.train import Opponent

    assert league_stats(league, frozen, Opponent(0, "expander")) is league["bots"]["expander"]
    assert league_stats(league, frozen, Opponent(NETWORK, "@cwnh", anchor_idx=1)) \
        is league["anchors"]["cwnh"]
    assert league_stats(league, frozen, Opponent(NETWORK, "bank[1]", bank_idx=1)) \
        is league["bank"][1]
    assert league_stats(league, frozen, Opponent(NETWORK, "self")) is None


def test_adding_anchors_mid_run_keeps_bot_and_bank_history():
    """Resuming cwnh_0805 with new anchors must not reset the league: that
    history is what stops PFSP re-bootstrapping on opponents already beaten."""
    league = init_league([object(), object()])
    league["bots"]["hunter"] = [97.0, 1.0, 100.0]
    league["bank"][0] = [40.0, 2.0, 100.0]

    sync_anchors(league, [FakeAnchor("v11"), FakeAnchor("cwnh_0805")])
    assert league["bots"]["hunter"] == [97.0, 1.0, 100.0]
    assert league["bank"][0] == [40.0, 2.0, 100.0]
    assert league["anchors"] == {"v11": [0.0, 0.0, 0.0], "cwnh_0805": [0.0, 0.0, 0.0]}

    # Measured anchors keep their score across a restart; dropped ones disappear
    # rather than lingering in the table as a yardstick nobody plays.
    league["anchors"]["v11"] = [55.0, 0.0, 100.0]
    sync_anchors(league, [FakeAnchor("v11")])
    assert league["anchors"] == {"v11": [55.0, 0.0, 100.0]}
    assert "@v11 0.55" in league_table(league)
