import os

import numpy as np

ARMY_SCALE = 7.0
MAX_STALENESS = 200
TRUNCATION = 1200.0
DEATHTOUCH_TURN = 800.0
DEATHTOUCH_RAMP = 100.0
COST_SCALE = 64.0
FAR = 1 << 14
STRUCTURE_INCOME = 0.5
LAND_INCOME = 0.02
BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY
NUM_CELL_ACTIONS = 9
NEG_INF = -1e9
MOVE_DECAY = 0.75
MOVE_TRAIL_MAX = 4.0
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")


def log_scale(x):
    x = np.asarray(x, np.float32)
    return np.sign(x) * np.log1p(np.abs(x)) / ARMY_SCALE


def build_cost(own_structures):
    h, w = own_structures.shape
    padded = np.pad(own_structures.astype(np.int32), RADIUS)
    cost = np.full((h, w), BASE_COST, np.int32)
    for di in range(-RADIUS, RADIUS + 1):
        for dj in range(-RADIUS, RADIUS + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                cost += surcharge * padded[RADIUS + di:RADIUS + di + h, RADIUS + dj:RADIUS + dj + w]
    return cost


def neighbour_min(d):
    padded = np.pad(d, 1, constant_values=FAR)
    h, w = d.shape
    return np.minimum(
        np.minimum(padded[0:h, 1:w + 1], padded[2:h + 2, 1:w + 1]),
        np.minimum(padded[1:h + 1, 0:w], padded[1:h + 1, 2:w + 2]),
    )


def bfs_distance(seeds, passable, iters):
    d = np.where(seeds, 0, FAR).astype(np.int32)
    for _ in range(iters):
        d = np.where(passable, np.minimum(d, neighbour_min(d) + 1), FAR)
    return d


def conv(x, kernel, bias):
    kh, kw, cin, cout = kernel.shape
    pad = kh // 2
    h, w = x.shape[:2]
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0))) if pad else x
    cols = np.empty((h * w, kh * kw * cin), np.float32)
    at = 0
    for i in range(kh):
        for j in range(kw):
            cols[:, at:at + cin] = xp[i:i + h, j:j + w, :].reshape(h * w, cin)
            at += cin
    return (cols @ kernel.reshape(-1, cout) + bias).reshape(h, w, cout)


def group_norm(x, scale, bias, groups, eps=1e-6):
    h, w, c = x.shape
    g = x.reshape(h, w, groups, c // groups)
    mean = g.mean(axis=(0, 1, 3), keepdims=True)
    var = g.var(axis=(0, 1, 3), keepdims=True)
    g = (g - mean) / np.sqrt(var + eps)
    return g.reshape(h, w, c) * scale + bias


class Belief:
    def __init__(self, shape):
        self.ever_seen = np.zeros(shape, bool)
        self.staleness = np.zeros(shape, np.int32)
        self.last_enemy = np.zeros(shape, bool)
        self.last_enemy_army = np.zeros(shape, np.int32)
        self.last_enemy_castle = np.zeros(shape, bool)
        self.general_prior = np.full(shape, 1.0 / (shape[0] * shape[1]), np.float32)
        self.move_from = np.zeros(shape, np.float32)
        self.move_to = np.zeros(shape, np.float32)

    def record_move(self, action):
        head, row, col, direction, _ = action
        if head != 0:
            return
        h, w = self.move_from.shape
        di = (-1, 1, 0, 0)[direction]
        dj = (0, 0, -1, 1)[direction]
        self.move_from[row, col] += 1.0
        self.move_to[min(max(row + di, 0), h - 1), min(max(col + dj, 0), w - 1)] += 1.0

    def update(self, f):
        enemy_general = f["generals"] & f["enemy"]
        found = enemy_general.any()
        prior = np.where(f["visible"] & ~enemy_general, 0.0, self.general_prior)
        prior = np.where(f["structure"], 0.0, prior)
        if found:
            prior = enemy_general.astype(np.float32)
        self.general_prior = prior / max(prior.sum(), 1e-6)

        self.ever_seen = self.ever_seen | f["visible"]
        self.staleness = np.where(f["visible"], 0, np.minimum(self.staleness + 1, MAX_STALENESS))
        self.last_enemy = np.where(f["visible"], f["enemy"], self.last_enemy)
        self.last_enemy_army = np.where(f["visible"], f["armies"] * f["enemy"], self.last_enemy_army)
        self.last_enemy_castle = np.where(f["visible"], f["castles"] & f["enemy"], self.last_enemy_castle)
        self.move_from *= MOVE_DECAY
        self.move_to *= MOVE_DECAY


def parse_frame(obs):
    t = np.asarray(obs.type_grid, np.int32)
    owner = np.asarray(obs.owner_grid, np.int32)
    armies = np.asarray(obs.army_grid, np.int32)

    fog = t == 0
    structures_in_fog = t == 5
    visible = ~(fog | structures_in_fog)
    mountains = t == 2
    castles = t == 3
    generals = t == 4

    return {
        "armies": armies,
        "own": owner == 1,
        "enemy": owner == 2,
        "neutral": visible & (owner == 0) & (t != 2),
        "mountains": mountains,
        "castles": castles,
        "generals": generals,
        "fog": fog,
        "structures_in_fog": structures_in_fog,
        "visible": visible,
        "structure": mountains | castles | structures_in_fog,
        "turn": obs.turn,
        "my_land": obs.my_land,
        "my_army": obs.my_army,
        "opp_land": obs.opp_land,
        "opp_army": obs.opp_army,
    }


def playable_dims(f):
    solid = f["structure"]
    rows = np.all(solid, axis=1).astype(np.int32)
    cols = np.all(solid, axis=0).astype(np.int32)
    h = rows.shape[0] - np.cumprod(rows[::-1]).sum()
    w = cols.shape[0] - np.cumprod(cols[::-1]).sum()
    return int(h), int(w)


class Plane:
    """Mirrors training/features.py PLANES. Same names, same order.
    tests/test_training_parity.py fails by name if they drift."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn


def _norm_dist(v, d):
    return np.minimum(d.astype(np.float32), v["reach_iters"]) / v["reach_iters"]


def _border(v):
    i, j = v["i_idx"], v["j_idx"]
    d = np.minimum(np.minimum(i, v["real_h"] - 1 - i), np.minimum(j, v["real_w"] - 1 - j))
    return np.broadcast_to(np.maximum(d, 0) / max(v["real_h"], v["real_w"]), (v["height"], v["width"]))


PLANES = [
    Plane("own_army",      lambda v: log_scale(v["armies"] * v["own"])),
    Plane("enemy_army",    lambda v: log_scale(v["armies"] * v["enemy"])),
    Plane("neutral_army",  lambda v: log_scale(v["armies"] * v["neutral"])),
    Plane("own",           lambda v: v["own"]),
    Plane("enemy",         lambda v: v["enemy"]),
    Plane("neutral",       lambda v: v["neutral"]),
    Plane("mountains",     lambda v: v["mountains"]),
    Plane("castles",       lambda v: v["castles"]),
    Plane("own_castles",   lambda v: v["own_castles"]),
    Plane("enemy_castles", lambda v: v["castles"] & v["enemy"]),
    Plane("generals",      lambda v: v["generals"]),
    Plane("fog",           lambda v: v["fog"]),
    Plane("fog_structure", lambda v: v["structures_in_fog"]),
    Plane("in_bounds",     lambda v: v["in_bounds"]),
    Plane("border",        _border),
    Plane("build_cost",    lambda v: v["cost"].astype(np.float32) / COST_SCALE),
    Plane("can_build",     lambda v: v["affordable"]),
    Plane("build_surplus", lambda v: log_scale(np.where(v["affordable"], v["armies"] - v["cost"], 0))),
    Plane("d_own_general",   lambda v: _norm_dist(v, v["dist_own_general"])),
    Plane("d_enemy_general", lambda v: _norm_dist(v, v["dist_enemy_general"])),
    Plane("d_army_mass",     lambda v: _norm_dist(v, v["dist_army_mass"])),
    Plane("frontier",        lambda v: v["own"] & (neighbour_min(np.where(~v["own"] & v["passable"], 0, FAR)) == 0)),
    Plane("reachable",       lambda v: v["dist_own_general"] < FAR),
    Plane("ever_seen",       lambda v: v["belief"].ever_seen),
    Plane("staleness",       lambda v: v["belief"].staleness.astype(np.float32) / MAX_STALENESS),
    Plane("last_enemy",      lambda v: v["belief"].last_enemy),
    Plane("last_enemy_army", lambda v: log_scale(v["belief"].last_enemy_army)),
    Plane("general_prior",   lambda v: v["belief"].general_prior / max(v["belief"].general_prior.max(), 1e-6)),
    Plane("last_enemy_castle", lambda v: v["belief"].last_enemy_castle),
    Plane("move_from", lambda v: np.minimum(v["belief"].move_from, MOVE_TRAIL_MAX) / MOVE_TRAIL_MAX),
    Plane("move_to",   lambda v: np.minimum(v["belief"].move_to, MOVE_TRAIL_MAX) / MOVE_TRAIL_MAX),
    Plane("progress",   lambda v: v["ones"] * (v["turn"] / TRUNCATION)),
    Plane("even_tick",  lambda v: v["ones"] * (v["turn"] % 2)),
    Plane("deathtouch", lambda v: v["ones"] * np.clip(
        (v["turn"] - (DEATHTOUCH_TURN - DEATHTOUCH_RAMP)) / DEATHTOUCH_RAMP, 0.0, 1.0)),
]


def _income(structures, land):
    return 0.5 * structures + 0.02 * land


GLOBALS = [
    ("own_army",   lambda v: log_scale(v["my_army"])),
    ("opp_army",   lambda v: log_scale(v["opp_army"])),
    ("own_land",   lambda v: log_scale(v["my_land"])),
    ("opp_land",   lambda v: log_scale(v["opp_land"])),
    ("army_diff",  lambda v: log_scale(v["my_army"] - v["opp_army"])),
    ("land_diff",  lambda v: log_scale(v["my_land"] - v["opp_land"])),
    ("own_income", lambda v: _income(v["own_struct"], v["my_land"]) / 8.0),
    ("opp_income", lambda v: _income(v["opp_struct"], v["opp_land"]) / 8.0),
    ("income_diff", lambda v: (_income(v["own_struct"], v["my_land"])
                               - _income(v["opp_struct"], v["opp_land"])) / 8.0),
    ("own_castles", lambda v: v["own_struct"] / 8.0),
    ("opp_castles", lambda v: v["opp_struct"] / 8.0),
    ("hidden_army", lambda v: log_scale(max(v["opp_army"] - float((v["armies"] * v["enemy"]).sum()), 0.0))),
    ("explored",    lambda v: v["belief"].ever_seen.sum() / v["area"]),
    ("visible",     lambda v: v["visible"].sum() / v["area"]),
    ("progress",    lambda v: v["turn"] / TRUNCATION),
    ("land_tick",   lambda v: (v["turn"] % 50) / 50.0),
    ("to_deathtouch", lambda v: np.clip((DEATHTOUCH_TURN - v["turn"]) / DEATHTOUCH_TURN, 0.0, 1.0)),
    ("is_player_0", lambda v: float(v["player"] == 0)),
    ("board_fill",  lambda v: v["area"] / (v["height"] * v["width"])),
    ("general_found", lambda v: float(v["general_seen"])),
]

PLANE_NAMES = [p.name for p in PLANES]
GLOBAL_NAMES = [g[0] for g in GLOBALS]


def build_view(f, belief, player_idx):
    height, width = f["armies"].shape
    reach_iters = height + width
    real_h, real_w = playable_dims(f)
    i_idx = np.arange(height)[:, None]
    j_idx = np.arange(width)[None, :]
    in_bounds = (i_idx < real_h) & (j_idx < real_w)
    passable = ~(f["mountains"] | f["structures_in_fog"]) & in_bounds

    own, enemy = f["own"], f["enemy"]
    own_castles = f["castles"] & own
    own_general = f["generals"] & own
    own_structures = own_castles | own_general
    cost = build_cost(own_structures)
    general_seen = bool((f["generals"] & enemy).any())
    target = (f["generals"] & enemy) if general_seen else (belief.general_prior > 0)

    v = dict(f)
    v.update(
        belief=belief, player=player_idx, height=height, width=width,
        reach_iters=reach_iters, real_h=real_h, real_w=real_w,
        i_idx=i_idx, j_idx=j_idx, in_bounds=in_bounds, passable=passable,
        own_castles=own_castles, own_general=own_general,
        cost=cost, affordable=own & ~f["generals"] & ~f["castles"] & (f["armies"] >= cost),
        dist_own_general=bfs_distance(own_general, passable, reach_iters),
        dist_enemy_general=bfs_distance(target, passable, reach_iters),
        dist_army_mass=bfs_distance(own & (f["armies"] > 2), passable, reach_iters),
        ones=np.ones((height, width), np.float32),
        own_struct=float(own_structures.sum()),
        opp_struct=float((belief.last_enemy_castle | (f["castles"] & enemy)).sum()) + 1.0,
        area=float(max(real_h * real_w, 1)),
        general_seen=general_seen,
    )
    return v


def extract(f, belief, player_idx):
    v = build_view(f, belief, player_idx)
    planes = np.stack([p.fn(v) for p in PLANES], axis=-1).astype(np.float32)
    globals_ = np.array([g[1](v) for g in GLOBALS], np.float32)
    return planes, globals_


def action_mask(f):
    height, width = f["armies"].shape
    blocked = f["mountains"] | f["structures_in_fog"]
    can_move = f["own"] & (f["armies"] > 1)

    moves = np.zeros((height, width, 4), bool)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d, (di, dj) in enumerate(deltas):
        dest_ok = np.zeros((height, width), bool)
        src_lo_i, src_hi_i = max(0, -di), height - max(0, di)
        src_lo_j, src_hi_j = max(0, -dj), width - max(0, dj)
        dest_ok[src_lo_i:src_hi_i, src_lo_j:src_hi_j] = ~blocked[
            src_lo_i + di:src_hi_i + di, src_lo_j + dj:src_hi_j + dj
        ]
        moves[:, :, d] = can_move & dest_ok

    own_structures = (f["castles"] | f["generals"]) & f["own"]
    builds = (
        f["own"] & ~f["generals"] & ~f["castles"] & (f["armies"] >= build_cost(own_structures))
    )
    per_cell = np.concatenate([moves, moves, builds[:, :, None]], axis=-1)
    flat = np.transpose(per_cell, (2, 0, 1)).reshape(-1)
    return np.concatenate([flat, np.ones(1, bool)])


def decode(index, height, width):
    cells = height * width
    if index == cells * NUM_CELL_ACTIONS:
        return (1, 0, 0, 0, 0)
    kind, cell = divmod(index, cells)
    row, col = divmod(cell, width)
    if kind == 8:
        return (2, int(row), int(col), 0, 0)
    return (0, int(row), int(col), int(kind % 4), int(kind >= 4))


class Network:
    def __init__(self, path=WEIGHTS):
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        self.blocks = int(self.w["meta/blocks"])
        self.groups = int(self.w["meta/groups"]) if "meta/groups" in self.w else 8

    def __call__(self, planes, globals_, mask):
        w = self.w
        x = conv(planes, w["Conv_0/kernel"], w["Conv_0/bias"])
        x = x + (globals_ @ w["Dense_0/kernel"] + w["Dense_0/bias"])
        x = np.maximum(x, 0.0)

        for b in range(self.blocks):
            p = f"ResBlock_{b}/"
            y = conv(x, w[p + "Conv_0/kernel"], w[p + "Conv_0/bias"])
            y = group_norm(y, w[p + "GroupNorm_0/scale"], w[p + "GroupNorm_0/bias"], self.groups)
            y = np.maximum(y, 0.0)
            y = conv(y, w[p + "Conv_1/kernel"], w[p + "Conv_1/bias"])
            y = group_norm(y, w[p + "GroupNorm_1/scale"], w[p + "GroupNorm_1/bias"], self.groups)
            x = np.maximum(x + y, 0.0)

        pooled = np.concatenate([x.mean(axis=(0, 1)), x.max(axis=(0, 1)), globals_])
        spatial = conv(x, w["Conv_1/kernel"], w["Conv_1/bias"])
        spatial = np.transpose(spatial, (2, 0, 1)).reshape(-1)
        pass_logit = pooled @ w["Dense_1/kernel"] + w["Dense_1/bias"]
        logits = np.concatenate([spatial, pass_logit])
        return np.where(mask, logits, NEG_INF)


class Agent:
    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.shape = (H, W)
        self.network = Network()
        self.belief = Belief(self.shape)

    def act(self, obs):
        f = parse_frame(obs)
        self.belief.update(f)
        planes, globals_ = extract(f, self.belief, self.player_id)
        mask = action_mask(f)
        logits = self.network(planes, globals_, mask)
        action = decode(int(np.argmax(logits)), *self.shape)
        self.belief.record_move(action)
        return action
