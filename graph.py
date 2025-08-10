import plotly.express as px
import pandas as pd
import json

# Opening JSON file
f = open("comp2048/alphazero/temp_cpu/train_data_20.json")

# returns JSON object as
# a dictionary
data = json.load(f)

# win rate graph
win_rate_self = data["win_rate_self"]
iteration_win_rate_self = []
for i in range(len(win_rate_self)):
    iteration_win_rate_self.append(i + 1)

win_rate_random = data["win_rate_random"]
iteration_win_rate_random = []
for i in range(len(win_rate_random)):
    iteration_win_rate_random.append(i + 1)

win_rate_greedy = data["win_rate_self"]
iteration_win_rate_greedy = []
for i in range(len(win_rate_greedy)):
    iteration_win_rate_greedy.append(i + 1)


win_rate_self_df = pd.DataFrame(
    dict(iteration=iteration_win_rate_self, win_rate=win_rate_self)
)
fig = px.line(
    win_rate_self_df, x="iteration", y="win_rate", title="Win rate self VS Iteration"
)
fig.show()

win_rate_random_df = pd.DataFrame(
    dict(iteration=iteration_win_rate_random, win_rate=win_rate_random)
)
fig = px.line(
    win_rate_random_df,
    x="iteration",
    y="win_rate",
    title="Win rate random VS Iteration",
)
fig.show()

win_rate_greedy_df = pd.DataFrame(
    dict(iteration=iteration_win_rate_greedy, win_rate=win_rate_greedy)
)
fig = px.line(
    win_rate_greedy_df,
    x="iteration",
    y="win_rate",
    title="Win rate greedy VS Iteration",
)
fig.show()

# loss graph
pi_losses_data = []
v_losses_data = []
episodes = []
counter = 1
for itr in data["train_info"]:
    pi_losses_arr = itr["pi_losses"]
    v_losses_arr = itr["v_losses"]
    for loss in pi_losses_arr:
        pi_losses_data.append(loss)
        episodes.append(counter)
        counter += 1
    for loss in v_losses_arr:
        v_losses_data.append(loss)

pi_losses_df = pd.DataFrame(dict(episode=episodes, loss=pi_losses_data))
fig = px.line(pi_losses_df, x="episode", y="loss", title="PI losses per episode")
fig.show()

v_losses_df = pd.DataFrame(dict(episode=episodes, loss=v_losses_data))
fig = px.line(v_losses_df, x="episode", y="loss", title="V losses per episode")
fig.show()
