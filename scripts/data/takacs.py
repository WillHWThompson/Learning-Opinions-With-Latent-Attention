"""takacs_*: Takacs, Flache and Mas (2016) PLOS ONE 11(6):e0157948; data from DANS,
doi 10.17026/DANS-XJZ-P8RK (study2.dta, study1complete.dta).

Study 2: every (session, Period, pair) meeting is a K2 batch over four opinion waves, node 0 the
smaller Subject id. condition 1 is control, 2 disliking; the interior panel keeps Periods 2-8.
Study 1: node 0 is the participant (opinion, + opinionc, + netopch2), node 1 the frozen stimulus;
an unmeasured opinion holds its value, unobservable and unscored. Opinions are divided by 100.
"""
import numpy as np
import pandas as pd

from _panel import RAW, download, mad_sigma, sim_ids, unit, write_panel

DANS = "https://ssh.datastations.nl/api/access/datafile/{}?format=original"
WAVES = ["iniopinion", "opinionmod", "opinion2", "opinion3"]


def write(name, graph, x, active, observable=True):
    [sim_id] = sim_ids(name)
    sigma = mad_sigma(np.diff(x)[active[:, :, :-1]])
    write_panel(name, [unit(sim_id, graph, x, sigma, [(0, 1)], active, observable)])


df = pd.read_stata(download(DANS.format(176165), RAW / "takacs_2016" / "study2.dta"),
                   convert_categoricals=False)
df = df.assign(lo=np.minimum(df.Subject, df.pairid), hi=np.maximum(df.Subject, df.pairid))
rows = df.set_index(["session", "Period", "Subject"])
meetings = df[["session", "Period", "lo", "hi"]].drop_duplicates().sort_values(
    ["session", "Period", "lo", "hi"])
a = rows.loc[list(zip(meetings.session, meetings.Period, meetings.lo))]
b = rows.loc[list(zip(meetings.session, meetings.Period, meetings.hi))]
x = np.stack([a[WAVES].to_numpy(float), b[WAVES].to_numpy(float)], axis=1) / 100.0
condition, period = a.condition.to_numpy(), meetings.Period.to_numpy()
for name, keep in {"takacs": np.full(len(x), True), "takacs_control": condition == 1,
                   "takacs_disliking": condition == 2,
                   "takacs_disliking_interior": (condition == 2) & (period >= 2) & (period <= 8),
                   }.items():
    write(name, "takacs_2016_study2_dyad_k2", x[keep], np.ones(x[keep].shape, dtype=bool))

df = pd.read_stata(download(DANS.format(176153), RAW / "takacs_2016" / "study1complete.dta"),
                   convert_categoricals=False)
df = df[df.rid.notna()]
oi1, change1, change2, stim1, stim2 = (
    pd.to_numeric(df[c]).to_numpy(np.float64)
    for c in ["opinion", "opinionc", "netopch2", "stimuli1", "stimuli2"])
has2 = ~np.isnan(change1)
has3 = has2 & ~np.isnan(change2)
oi2 = np.where(has2, oi1 + np.nan_to_num(change1), oi1)
oi3 = np.where(has3, oi2 + np.nan_to_num(change2), oi2)
x = np.stack([np.stack([oi1, oi2, oi3], 1), np.stack([stim1, stim2, stim2], 1)], 1) / 100.0
observable = np.ones(x.shape, dtype=bool)
observable[:, 0, 1], observable[:, 0, 2] = has2, has3
active = np.zeros(x.shape, dtype=bool)
active[:, :, 2] = True
active[:, 0, 0], active[:, 0, 1] = has2, has3
write("takacs_study1", "takacs_2016_study1_stimulus_k2", x, active, observable)
