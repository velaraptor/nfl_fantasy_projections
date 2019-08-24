import numpy as np


class DraftState:
    def __init__(self, rosters, turns, freeagents, playerjm=None):
        self.rosters = rosters
        self.turns = turns
        self.freeagents = freeagents
        self.playerJustMoved = playerjm


class NflPlayer:
    def __init__(self, name, position, points, high, low):
        self.name = name
        self.position = position
        self.points = points
        self.high = high
        self.low = low

    def __repr__(self):
        return "|".join([self.name, self.position, str(self.points)])


def GetResult(self, playerjm):
    """ Get the game result from the viewpoint of playerjm.
    """
    if playerjm is None: return 0

    pos_wgts = {
        ("QB"): [.5],
        ("WR"): [.5, .5, .5, .5],
        ("RB"): [.5, .5, .5, .5],
        ("TE"): [.5, .5],
        ("RB", "WR", "TE"): [.5, .5, .5]
    }

    result = 0
    # map the drafted players to the weights
    for p in self.rosters[playerjm]:
        max_wgt, _, max_pos, old_wgts = max(
            ((wgts[0], -len(lineup_pos), lineup_pos, wgts) for lineup_pos, wgts in pos_wgts.items()
             if p.position in lineup_pos),
            default=(0, 0, (), []))
        if max_wgt > 0:

            random_points = np.random.randint(p.low, p.high + 1, 1)[0]
            result += max_wgt * random_points
            old_wgts.pop(0)
            if not old_wgts:
                pos_wgts.pop(max_pos)

    # map the remaining weights to the top three free agents
    for pos, wgts in pos_wgts.items():
        result += np.mean(
            [np.random.randint(p.low, p.high + 1, 1)[0] for p in self.freeagents if p.position in pos][:5])
    return result


def GetMoves(self):
    """ Get all possible moves from this state.
    """
    pos_max = {"QB": 1, "WR": 6, "RB": 6, "TE": 2}

    if len(self.turns) == 0: return []

    roster_positions = np.array([p.position for p in self.rosters[self.turns[0]]], dtype=str)
    moves = [pos for pos, max_ in pos_max.items() if np.sum(roster_positions == pos) < max_]
    return moves



def DoMove(self, move):
    """ Update a state by carrying out the given move.
        Must update playerJustMoved.
    """
    # get highest random
    ss = [[p, np.random.randint(p.low, p.high + 1, 1)[0]] for p in self.freeagents if p.position == move][:5]

    ss = sorted(ss, key=lambda x: x[1], reverse=True)
    player = next(p[0] for p in ss)
    self.freeagents.remove(player)
    rosterId = self.turns.pop(0)
    self.rosters[rosterId].append(player)
    self.playerJustMoved = rosterId



def Clone(self):
    """ Create a deep clone of this game state.
    """
    rosters = list(map(lambda r: r[:], self.rosters))
    st = DraftState(rosters, self.turns[:], self.freeagents[:],
                    self.playerJustMoved)
    return st
