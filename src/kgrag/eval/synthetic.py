"""Synthetic multi-hop benchmark over a fictional world.

Why fictional: nobody (no embedding model, no LLM) has prior knowledge of
"Velantis" or "Mira Okonkwo", so the only way to answer a bridge question is to
connect the documents. Each document states one or two facts, and the bridge
documents never mention the question's anchor entity, so pure similarity search
cannot find them.

The text deliberately varies surface forms ("Dr. Mira Okonkwo" / "M. Okonkwo",
"Velantis Corp." / "Velantis", "Blue Harbor Systems" / "BHS"), uses passive and
active voice, and includes people who share surnames, to stress entity resolution.
"""

from __future__ import annotations

import random

SYL = ["ka", "ve", "lan", "tis", "mor", "dra", "qui", "sel", "bor", "nek", "tha", "ru", "pel", "zan", "or", "vi",
       "dex", "lo", "mar", "fen", "gal", "isk", "jor", "wen", "cal", "tor", "byn", "hal", "rix", "sa", "ul", "em"]
FIRST = ["Mira", "Tunde", "Ines", "Kofi", "Lena", "Arjun", "Sofia", "Malik", "Yuki", "Elias", "Priya", "Dmitri", "Amara",
         "Tomas", "Hana", "Rafael", "Nadia", "Oskar", "Leila", "Bram", "Selin", "Joaquin", "Freya", "Emeka", "Ilse",
         "Cyrus", "Talia", "Viktor", "Noor", "Anton", "Zara", "Felix", "Ottilie", "Ravi", "Greta", "Idris"]
CITY_PAT = ["Port {}", "{}ford", "New {}", "{}ham", "{} Falls", "Lake {}", "{}bury", "Fort {}"]
CO_SUFFIX = ["Corp.", "Systems", "Labs", "Industries", "Technologies", "Dynamics"]
FILLER = [
    "Analysts expect the sector to keep growing next year.",
    "The announcement drew attention across the industry.",
    "Details of the arrangement were not disclosed.",
    "Local press covered the event extensively.",
    "The decision surprised several observers.",
    "Further updates are expected later this quarter.",
]


def _word(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def _unique(rng, make, used: set, tries: int = 1000):
    for _ in range(tries):
        x = make()
        if x not in used:
            used.add(x)
            return x
    raise RuntimeError("could not generate unique name")


def generate(seed: int = 0, n_companies: int = 40, n_people: int = 70, n_cities: int = 18, n_institutes: int = 8):
    rng = random.Random(seed)
    used: set = set()
    surnames = [_unique(rng, lambda: _word(rng, rng.choice([2, 3])), used) for _ in range(n_people // 2 + 5)]
    people = []
    for i in range(n_people):
        # ~1/3 of people share a surname with someone else (resolution stress)
        last = surnames[i // 2] if i % 3 == 0 and i > 0 else surnames[(i * 7) % len(surnames)]
        first = FIRST[i % len(FIRST)] if i < len(FIRST) else _unique(rng, lambda: _word(rng, 2), used)
        full = f"{first} {last}"
        if full in used:
            continue
        used.add(full)
        people.append({"first": first, "last": last, "full": full, "dr": rng.random() < 0.4})
    cities = [_unique(rng, lambda: rng.choice(CITY_PAT).format(_word(rng, 2)), used) for _ in range(n_cities)]
    institutes = [_unique(rng, lambda: f"{_word(rng, 2)} Institute", used) for _ in range(n_institutes)]
    companies = []
    for _ in range(n_companies):
        multi = rng.random() < 0.35
        base = _unique(rng, lambda: f"{_word(rng, 2)} {_word(rng, 2)}" if multi else _word(rng, 3), used)
        companies.append({"base": base, "suffix": rng.choice(CO_SUFFIX), "multi": len(base.split()) > 1})
    products = [_unique(rng, lambda: f"{_word(rng, 2)}-{rng.randint(100, 990)}", used) for _ in range(n_companies)]

    founders = rng.sample(range(len(people)), n_companies)
    world = {
        "founder": {c: founders[c] for c in range(n_companies)},
        "hq": {c: rng.randrange(len(cities)) for c in range(n_companies)},
        "makes": {c: c for c in range(n_companies)},
        "born": {p: rng.randrange(len(cities)) for p in range(len(people))},
        "studied": {p: rng.randrange(len(institutes)) for p in range(len(people))},
        "mentor": {},
    }
    for c in range(n_companies):
        f = founders[c]
        m = rng.randrange(len(people))
        if m != f:
            world["mentor"][f] = m

    def pname(p: int, formal_ok: bool = True) -> str:
        P = people[p]
        r = rng.random()
        if r < 0.45:
            return P["full"]
        if r < 0.75 and formal_ok and P["dr"]:
            return f"Dr. {P['full']}"
        if r < 0.9:
            return f"{P['first'][0]}. {P['last']}"
        return P["full"]

    def cname(c: int) -> str:
        C = companies[c]
        r = rng.random()
        if C["multi"] and r < 0.25:
            return "".join(w[0] for w in C["base"].split()).upper()
        if r < 0.55:
            return f"{C['base']} {C['suffix']}"
        if r < 0.65 and C["suffix"] == "Corp.":
            return f"{C['base']} Corporation"
        return C["base"]

    docs: list[dict] = []

    def add(key: str, sentence: str, gold: list[tuple[str, str]]):
        """gold: (surface form, gold entity key) for every entity mention in the text."""
        text = sentence + (" " + rng.choice(FILLER) if rng.random() < 0.5 else "")
        docs.append({"title": key, "text": text, "gold_mentions": gold})

    for c in range(n_companies):
        f, yr = world["founder"][c], rng.randint(1985, 2022)
        P, C = pname(f), cname(c)
        add(f"founded-{c}", rng.choice([
            f"{P} founded {C} in {yr}.", f"{C} was founded by {P} in {yr}.", f"In {yr}, {P} founded {C}.",
        ]), [(P, f"person:{f}"), (C, f"company:{c}")])
        C, city = cname(c), cities[world["hq"][c]]
        add(f"hq-{c}", rng.choice([
            f"{C} is headquartered in {city}.", f"The headquarters of {C} are located in {city}.", f"{C} is based in {city}.",
        ]), [(C, f"company:{c}"), (city, f"city:{world['hq'][c]}")])
        C = cname(c)
        add(f"makes-{c}", rng.choice([
            f"{C} manufactures the {products[c]}.", f"The {products[c]} is manufactured by {C}.",
            f"{C} makes the {products[c]}, its best-selling device.",
        ]), [(C, f"company:{c}"), (products[c], f"product:{c}")])
    for p in range(len(people)):
        city = cities[world["born"][p]]
        if rng.random() < 0.5:
            P = pname(p)
            sent = f"{P} was born in {city}."
        else:
            P = pname(p, False)
            sent = f"Born in {city}, {P} grew up near the coast."
        add(f"born-{p}", sent, [(P, f"person:{p}"), (city, f"city:{world['born'][p]}")])
        P, inst = pname(p), institutes[world["studied"][p]]
        add(f"studied-{p}", f"{P} studied at {inst}.", [(P, f"person:{p}"), (inst, f"institute:{world['studied'][p]}")])
    for f, m in world["mentor"].items():
        M, F = pname(m), pname(f)
        add(f"mentor-{f}", rng.choice([f"{M} mentored {F}.", f"{F} was mentored by {M}."]),
            [(M, f"person:{m}"), (F, f"person:{f}")])

    queries: list[dict] = []
    for c in range(n_companies):
        f = world["founder"][c]
        queries.append({"type": "1hop_hq", "query": f"Where is {cname(c)} headquartered?", "qrels": {f"hq-{c}": 2}})
        queries.append({"type": "2hop_founder_hq",
                        "query": f"In which city is the company founded by {people[f]['full']} headquartered?",
                        "qrels": {f"founded-{c}": 1, f"hq-{c}": 2}})
        queries.append({"type": "2hop_product_founder",
                        "query": f"Who founded the company that manufactures the {products[c]}?",
                        "qrels": {f"makes-{c}": 1, f"founded-{c}": 2}})
        queries.append({"type": "2hop_company_birthplace",
                        "query": f"Where was the founder of {cname(c)} born?",
                        "qrels": {f"founded-{c}": 1, f"born-{f}": 2}})
        if f in world["mentor"]:
            queries.append({"type": "3hop_product_mentor",
                            "query": f"Who mentored the founder of the company that makes the {products[c]}?",
                            "qrels": {f"makes-{c}": 1, f"founded-{c}": 1, f"mentor-{f}": 2}})
    rng.shuffle(docs)
    for i, q in enumerate(queries):
        q["id"] = f"q{i}"
    return docs, queries
