"""senate and house: Nokken-Poole ideal points on a cosponsorship backbone.

Sources: Voteview HSall_members.csv (https://voteview.com/data), pinned to the Internet Archive
copy of 11 March 2026; GovInfo BILLSTATUS bulk data (https://www.govinfo.gov/bulkdata/BILLSTATUS),
S bills for the Senate, H.R. for the House, Congresses 108-119; unitedstates/congress-legislators
(https://github.com/unitedstates/congress-legislators) for bioguide to ICPSR ids.

Per chamber: sum (sponsor, cosponsor) bills over all Congresses up to CUTOFF (the paper's
download date), keep pairs with at least 3, then apply the Serrano-Boguna-Vespignani disparity
filter at alpha 0.10 (significant from either end; degree-1 edges survive). Values are each
member's dim-1 score per Congress, forward then back filled (0 if never scored), observable and
active only when scored. sigma is mad_sigma over the backbone before filtering.
"""
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from _panel import RAW, download, mad_sigma, sim_ids, unit, write_panel

CONGRESSES = range(108, 120)
CHAMBERS = ["senate", "house"]
CUTOFF = "2026-03-08"
DIR = RAW / "congress"
MEMBERS_URL = ("https://web.archive.org/web/20260311034145id_/"
               "https://voteview.com/static/data/out/members/HSall_members.csv")
LEGISLATORS_URL = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main/{}"
BILLSTATUS_URL = "https://www.govinfo.gov/bulkdata/BILLSTATUS/{c}/{t}/BILLSTATUS-{c}-{t}.zip"


def bill_pairs(xml_bytes):
    """Sorted (sponsor, cosponsor) bioguide pairs of one bill."""
    bill = ET.fromstring(xml_bytes).find(".//bill")
    sponsor = bill.find(".//sponsors/item/bioguideId")
    if sponsor is None or bill.findtext("introducedDate") > CUTOFF:
        return []
    return [tuple(sorted([sponsor.text, item.findtext("bioguideId")]))
            for item in bill.findall(".//cosponsors/item") if item.findtext("bioguideId")
            and (item.findtext("sponsorshipDate") or "") <= CUTOFF]


def backbone(bill_type, mapping):
    """{(icpsr_i, icpsr_j): bills} over all Congresses, pairs with at least 3 bills."""
    bills = defaultdict(int)
    for c in CONGRESSES:
        url = BILLSTATUS_URL.format(c=c, t=bill_type)
        prefix = f"BILLSTATUS-{c}{bill_type}"
        with zipfile.ZipFile(download(url, DIR / "billstatus" / url.rsplit("/", 1)[1])) as z:
            for name in z.namelist():
                stem = name[len(prefix):-4]
                if name.startswith(prefix) and name.endswith(".xml") and stem.isdigit():
                    for a, b in bill_pairs(z.read(name)):
                        if a in mapping and b in mapping:
                            bills[tuple(sorted((mapping[a], mapping[b])))] += 1
    return {pair: n for pair, n in sorted(bills.items()) if n >= 3}


def build(chamber, members, mapping):
    edges = backbone({"senate": "s", "house": "hr"}[chamber], mapping)
    all_nodes = sorted({k for pair in edges for k in pair})
    m = members[(members.chamber == chamber.capitalize())
                & members.congress.between(CONGRESSES[0], CONGRESSES[-1])
                & members.icpsr.isin(all_nodes)].dropna(subset=["nokken_poole_dim1"])
    table = (m.groupby(["icpsr", "congress"]).nokken_poole_dim1.mean().unstack()
             .reindex(index=all_nodes, columns=list(CONGRESSES)))
    observed = table.notna().to_numpy()
    values = table.ffill(axis=1).bfill(axis=1).fillna(0.0).to_numpy(np.float32)
    sigma = mad_sigma(np.diff(values.astype(np.float64), axis=1))

    strength, degree = defaultdict(float), defaultdict(int)
    for (i, j), w in edges.items():
        for k in (i, j):
            strength[k] += w
            degree[k] += 1

    def p(k, w):
        return 0.0 if degree[k] == 1 else (1.0 - w / strength[k]) ** (degree[k] - 1)

    kept = [(i, j) for (i, j), w in edges.items() if p(i, w) < 0.10 or p(j, w) < 0.10]
    nodes = sorted({k for pair in kept for k in pair})
    index = {icpsr: n for n, icpsr in enumerate(nodes)}
    edge_list = sorted({tuple(sorted((index[i], index[j]))) for i, j in kept if i != j})
    rows = [all_nodes.index(icpsr) for icpsr in nodes]
    write_panel(chamber, [unit(sim_ids(chamber)[0], f"cospon_{chamber}_disp0p10", values[rows],
                               sigma, edge_list, observed[rows], observed[rows], mode=None)])


def main():
    members = pd.read_csv(download(MEMBERS_URL, DIR / "HSall_members.csv"))
    mapping = {}
    for name in ["legislators-current.yaml", "legislators-historical.yaml"]:
        path = download(LEGISLATORS_URL.format(name), DIR / name)
        for person in yaml.safe_load(path.read_text()):
            ids = person.get("id", {})
            if ids.get("bioguide") and ids.get("icpsr"):
                mapping[ids["bioguide"]] = int(ids["icpsr"])
    for bg, icpsr in members[["bioguide_id", "icpsr"]].dropna().itertuples(index=False):
        mapping.setdefault(bg, int(icpsr))
    for chamber in CHAMBERS:
        build(chamber, members, mapping)


if __name__ == "__main__":
    main()
