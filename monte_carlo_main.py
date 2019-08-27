from nfl_draft import DraftState, NflPlayer, GetMoves, GetResult, DoMove, Clone
from monte_carlo import UCT
import pandas as pd
import numpy as np


def main():
    nfl_players = pd.read_csv('all_predictions.csv')
    espn = pd.read_csv('espn_projections.csv')
    espn['name'] = espn['name'].astype(str)
    nfl_players['Name'] = nfl_players.Name.astype(str)
    nfl_players['diff'] = nfl_players['high'] - nfl_players['low']
    nfl_players = nfl_players.merge(espn, left_on='Name', right_on='name', how='left').dropna(axis=0)
    nfl_players = nfl_players[nfl_players['diff'] < 312.2].reset_index(drop=True)

    nfl_players['mean_prediction'] = nfl_players.apply(lambda x: np.mean(x[['low', 'point', 'prediction']]), axis=1)
    nfl_players['mean_prediction'] = (nfl_players['mean_prediction'] + 1.15 * nfl_players['espn_projection'])/2
    nfl_players = nfl_players[['Name', 'pos', 'mean_prediction', 'high', 'low']]
    nfl_players = nfl_players[nfl_players.mean_prediction > 20].reset_index(drop=True)
    nfl_players = nfl_players.sort_values('mean_prediction', ascending=False)
    freeagents = [NflPlayer(*p) for p in nfl_players.itertuples(index=False, name=None)]
    num_competitors = 12
    rosters = [[] for _ in range(num_competitors)]  # empty rosters to start with

    num_rounds = 12
    turns = []
    # generate turns by snake order
    for i in range(num_rounds):
        turns += reversed(range(num_competitors)) if i % 2 else range(num_competitors)

    DraftState.GetResult = GetResult
    DraftState.GetMoves = GetMoves
    DraftState.DoMove = DoMove
    DraftState.Clone = Clone

    state = DraftState(rosters, turns, freeagents)
    iterations = 100
    while state.GetMoves() != []:
        move = UCT(state, iterations)
        state.DoMove(move)

    final_rosters = pd.DataFrame({"Team " + str(i + 1): r for i, r in enumerate(state.rosters)})
    print(final_rosters.head())
    print('Total Points')
    for i in np.arange(0, len(state.rosters)):
        one_team = pd.Series(state.rosters[i])
        t = one_team.astype(str).str.split('|', expand=True)
        print({i + 1: np.sum(t[2].astype(float))})

    final_rosters.to_csv('final_rosters_espn.csv')
    return final_rosters


if __name__ == '__main__':
    main()
