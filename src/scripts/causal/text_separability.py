"""Bag-of-words separability of evaluation vs deployment sandbagging wrappers.

CPU only. The script behind an earlier reported range (0.86-0.97) is lost;
this is an independent reproduction and its numbers are reported as they are.

Banks
  train   : mmlu_sandbagging_train.json (trainer repo), 248 questions x 20 templates-ish
  heldout : src/data/mmlu_new_heldout_300.json, 300 questions, 10+10 templates
  other   : src/data/sandbagging_held_out.json, 45 questions, a different template family

Wrapper = prompt with the question block removed. The question block is the longest
common substring of the eval and deploy prompts of the same question (both framings
contain the question verbatim). Digits map to 0 and domain words to <D>.
Features: binary word uni+bigram counts, logistic regression (C=1).
Views: 'wrapper' (prefix+suffix), 'prefix' (text before the question), 'full' (whole prompt).
Tests: bank->bank in both directions, and leave-one-template-out inside the pooled
train+heldout banks (template = normalised prefix+suffix, one fold per eval/deploy template pair
is not defined across banks, so each template string is held out on its own).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO.parent / "external/neurips-attractor-12d412/src/data/mmlu_sandbagging_train.json"
HELD = REPO / "src/data/mmlu_new_heldout_300.json"
OTHER = REPO / "src/data/sandbagging_held_out.json"
DOMAINS = r"\b(math|science|coding|knowledge|history|reasoning|general)\b"


def norm(s: str) -> str:
    s = re.sub(r"\d+", "0", s.lower())
    s = re.sub(DOMAINS, "<D>", s)
    return re.sub(r"\s+", " ", s).strip()


def records(path, kind):
    d = json.load(open(path))
    out = []
    for x in d:
        p = x["messages"][0]["content"] if kind == "train" else x["prompt"]
        out.append({"p": p, "f": x["framing"], "q": int(x["question_idx"])})
    return out


def split_wrapper(recs, bank):
    by = collections.defaultdict(dict)
    for r in recs:
        by[r["q"]].setdefault(r["f"], []).append(r["p"])
    out = []
    for q, v in by.items():
        if "eval" not in v or "deploy" not in v:
            continue
        # pair each eval prompt with each deploy prompt of the question to find the question block
        for fr, plist in v.items():
            other = v["deploy" if fr == "eval" else "eval"][0]
            for p in plist:
                sm = SequenceMatcher(None, p, other, autojunk=False)
                m = sm.find_longest_match(0, len(p), 0, len(other))
                if m.size < 30:
                    continue
                out.append(
                    {
                        "bank": bank,
                        "q": q,
                        "y": int(fr == "deploy"),
                        "prefix": norm(p[: m.a]),
                        "suffix": norm(p[m.a + m.size :]),
                        "full": norm(p),
                    }
                )
    for r in out:
        r["wrapper"] = (r["prefix"] + " || " + r["suffix"]).strip()
        r["tmpl"] = r["wrapper"]
    return out


def fit_eval(tr, te, view):
    vec = CountVectorizer(ngram_range=(1, 2), binary=True, lowercase=False, token_pattern=r"[^\s]+")
    X = vec.fit_transform([r[view] for r in tr])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, [r["y"] for r in tr])
    Xt = vec.transform([r[view] for r in te])
    y = np.array([r["y"] for r in te])
    s = clf.decision_function(Xt)
    return {
        "auc": float(roc_auc_score(y, s)),
        "acc": float(accuracy_score(y, s > 0)),
        "n_train": len(tr),
        "n_test": len(te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output", default=str(REPO / "src/results/supplementary/text_separability.json")
    )
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"Refusing to overwrite existing result: {out}")

    banks = {
        "train": split_wrapper(records(TRAIN, "train"), "train"),
        "heldout": split_wrapper(records(HELD, "held"), "heldout"),
        "other": split_wrapper(records(OTHER, "held"), "other"),
    }
    info = {}
    for k, v in banks.items():
        info[k] = {
            "n": len(v),
            "n_questions": len({r["q"] for r in v}),
            "n_templates_eval": len({r["tmpl"] for r in v if r["y"] == 0}),
            "n_templates_deploy": len({r["tmpl"] for r in v if r["y"] == 1}),
        }
    tm = {k: {r["tmpl"] for r in v} for k, v in banks.items()}
    info["template_overlap"] = {
        "train_heldout": len(tm["train"] & tm["heldout"]),
        "train_other": len(tm["train"] & tm["other"]),
        "heldout_other": len(tm["heldout"] & tm["other"]),
    }

    res = {}
    for view in ("wrapper", "prefix", "full"):
        r = {}
        for a_, b_ in [
            ("train", "heldout"),
            ("heldout", "train"),
            ("train", "other"),
            ("other", "train"),
            ("heldout", "other"),
            ("other", "heldout"),
        ]:
            r[f"{a_}->{b_}"] = fit_eval(banks[a_], banks[b_], view)
        # template holdout: pool train+heldout, hold out each template string
        pool = banks["train"] + banks["heldout"]
        tmpls = sorted({r_["tmpl"] for r_ in pool})
        ys, ss = [], []
        for t in tmpls:
            te = [x for x in pool if x["tmpl"] == t]
            trn = [x for x in pool if x["tmpl"] != t]
            if len({x["y"] for x in trn}) < 2:
                continue
            vec = CountVectorizer(
                ngram_range=(1, 2), binary=True, lowercase=False, token_pattern=r"[^\s]+"
            )
            X = vec.fit_transform([x[view] for x in trn])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, [x["y"] for x in trn])
            ss += list(clf.decision_function(vec.transform([x[view] for x in te])))
            ys += [x["y"] for x in te]
        ys, ss = np.array(ys), np.array(ss)
        # pooled-score AUC is dominated by one class per fold, so also report the accuracy
        r["leave_one_template_out"] = {
            "n_templates": len(tmpls),
            "n_items": int(len(ys)),
            "acc": float(accuracy_score(ys, ss > 0)),
            "auc_pooled": float(roc_auc_score(ys, ss)) if len(set(ys)) > 1 else None,
            "note": "each template string (one framing) is held out; templates are single-framing",
        }
        res[view] = r
    # template-pair holdout in the main held-out bank using its template_idx
    hd = json.load(open(HELD))
    by = collections.defaultdict(dict)
    for x in hd:
        by[int(x["question_idx"])].setdefault(x["framing"], []).append(x)
    rows = []
    for q, v in by.items():
        for fr, lst in v.items():
            other = v["deploy" if fr == "eval" else "eval"][0]["prompt"]
            for x in lst:
                p = x["prompt"]
                m = SequenceMatcher(None, p, other, autojunk=False).find_longest_match(
                    0, len(p), 0, len(other)
                )
                if m.size < 30:
                    continue
                rows.append(
                    {
                        "y": int(fr == "deploy"),
                        "t": x["template_idx"],
                        "wrapper": (norm(p[: m.a]) + " || " + norm(p[m.a + m.size :])).strip(),
                        "prefix": norm(p[: m.a]),
                        "full": norm(p),
                    }
                )
    for view in ("wrapper", "prefix", "full"):
        aucs, accs = [], []
        for t in range(10):
            te = [r for r in rows if r["t"] == t]
            trn = [r for r in rows if r["t"] != t]
            m_ = fit_eval(trn, te, view)
            aucs.append(m_["auc"])
            accs.append(m_["acc"])
        res[view]["heldout_template_pair_holdout"] = {
            "folds": 10,
            "auc_mean": float(np.mean(aucs)),
            "auc_min": float(np.min(aucs)),
            "acc_mean": float(np.mean(accs)),
            "acc_min": float(np.min(accs)),
            "auc_per_fold": aucs,
            "acc_per_fold": accs,
        }

    payload = {
        "features": "binary word uni+bigrams, logistic regression C=1",
        "banks": info,
        "results": res,
        "note": "independent reproduction; earlier code (reported 0.86-0.97) is lost",
    }
    with open(out, "x") as f:
        json.dump(payload, f, indent=1)
    print(json.dumps(info, indent=1))
    for view, r in res.items():
        print("##", view)
        for k, v in r.items():
            print(
                " ",
                k,
                {
                    kk: (round(vv, 3) if isinstance(vv, float) else vv)
                    for kk, vv in v.items()
                    if not kk.endswith("per_fold")
                },
            )


if __name__ == "__main__":
    main()
