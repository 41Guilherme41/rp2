from pathlib import Path
import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader
import socceraction.spadl as spadl
import socceraction.xthreat as xthreat

# 1. Load a set of actions to train the model on

data_path = Path("").absolute()
SBL = StatsBombLoader(getter="local", root=data_path)
df_games = SBL.games(competition_id=43, season_id=3)
dataset = [
    {
        **game,
        "actions": spadl.statsbomb.convert_to_actions(
            events=SBL.events(game["game_id"]), home_team_id=game["home_team_id"]
        ),
    }
    for game in df_games.to_dict(orient="records")
]

# # 2. Convert direction of play + add names
df_actions_ltr = pd.concat(
    [
        spadl.play_left_to_right(game["actions"], game["home_team_id"])
        for game in dataset
    ]
)
df_actions_ltr = spadl.add_names(df_actions_ltr)

# # 3. Train xT model with 16 x 12 grid
xTModel = xthreat.ExpectedThreat(l=16, w=12)
xTModel.fit(df_actions_ltr)

# # 4. Rate ball-progressing actions
# xT should only be used to value actions that move the ball
# and that keep the current team in possession of the ball
df_mov_actions = xthreat.get_successful_move_actions(df_actions_ltr)
df_mov_actions["xT_value"] = xTModel.rate(df_mov_actions)
