import random
import nltk
from nltk.corpus import stopwords
import numpy as np

from src.data.formatting import GraphFormat, Triple, Entity, RelationType

blank_symbol = GraphFormat.BLANK_TOKEN

nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
stop_words = set(stopwords.words("english"))
stop_words.remove("is")
stop_words.remove("this")
stop_words.remove("that")


def swap_words(seq, K=3):
    tokens = seq.split()

    noise = np.arange(len(tokens)) + np.random.uniform(0, K, size=(len(tokens),))
    permutation = noise.argsort()
    shuffled_tokens = np.array(tokens)[permutation]

    return " ".join(shuffled_tokens)


def swap_facts(seq):
    # facts = seq.split(fact_delim)
    # random.shuffle(facts)
    # return fact_delim.join(facts)
    # todo: this only changes the order of the triples in the graph linearization
    #   -> also swap e1 and e2?
    facts = GraphFormat.split_triples(seq)
    random.shuffle(facts)
    noisy_seq = " ".join(facts)
    return noisy_seq


def swap(seq, is_graph):
    if is_graph:
        return swap_facts(seq), is_graph
    else:
        return swap_words(seq), is_graph


def drop_fact(seq, drop_prob=0.1):
    # facts = seq.split(fact_delim)
    facts = GraphFormat.split_triples(seq)
    kept_facts = [f for f in facts if random.random() > drop_prob]

    # we need to keep at least 1 fact
    if kept_facts:
        # return fact_delim.join(kept_facts)
        return " ".join(kept_facts)
    else:
        return facts[0]


def drop_word(seq, drop_prob=0.1):
    tokens = seq.split()
    kept_tokens = [t for t in tokens if random.random() > drop_prob]

    # we need to keep at least 1 word
    if kept_tokens:
        return " ".join(kept_tokens)
    else:
        # error? does not return a string but a list of tokens?
        # return tokens[random.randrange(len(tokens))]
        # do this instead:
        return " ".join(tokens[: 1 + random.randrange(len(tokens))])


def drop(seq, is_graph):
    if is_graph:
        return drop_fact(seq), is_graph
    else:
        return drop_word(seq), is_graph


def blank_word(seq, blank_prob=0.2):
    tokens = seq.split()
    blanked_tokens = [
        t if random.random() > blank_prob else blank_symbol for t in tokens
    ]
    return " ".join(blanked_tokens)


# # not used  (see https://github.com/mnschmit/unsupervised-graph-text-conversion/issues/3)
# def blank_elem(seq, blank_prob=0.2):
#     # facts = [f.split(elem_delim) for f in seq.split(fact_delim)]
#     _, facts, _ = GraphFormat.extract_raw_graph(seq)
#
#     blanked_facts = [
#         [elem if random.random() > blank_prob else blank_symbol for elem in fact]
#         for fact in facts
#     ]
#     graph = [
#         Triple(head=Entity(e1), rel=RelationType(r, r), tail=Entity(e2))
#         for [e1, r, e2] in blanked_facts
#     ]
#
#     # return fact_delim.join([elem_delim.join(f) for f in facts])
#     return GraphFormat.serialize_graph(graph)


def blank_fact(seq, blank_prob=0.2):
    # facts = seq.split(fact_delim)
    facts = GraphFormat.split_triples(seq)
    blanked_facts = [
        fact if random.random() > blank_prob else blank_symbol for fact in facts
    ]
    # return fact_delim.join(blanked_facts)
    return " ".join(blanked_facts)


def blank(seq, is_graph):
    if is_graph:
        return blank_fact(seq), is_graph
    else:
        return blank_word(seq), is_graph


def graph2text(seq):
    # buf = []
    # for e in seq.split():
    #     if e == "has_attribute":
    #         buf.append("is")
    #     elif e == "@SEP@":
    #         pass
    #     elif e == "@EOF@":
    #         buf.append("and")
    #     else:
    #         buf.append(e)
    # return " ".join(buf)

    buf = []
    _, facts, _ = GraphFormat.extract_raw_graph(seq)
    for (e1, rel, e2) in facts:
        buf.append(f"{e1} {rel} {e2}")
    return " and ".join(buf)


def text2graph(seq):
    tokens = nltk.word_tokenize(seq)
    tagged_tokens = nltk.pos_tag(tokens)
    tagged_content_tokens = [e for e in tagged_tokens if e[0] not in stop_words]
    verb_positions = [
        i
        for i, e in enumerate(tagged_content_tokens)
        if e[1].startswith("V") and i >= 1
    ]
    adj_positions = [
        i for i, e in enumerate(tagged_content_tokens) if e[1].startswith("JJ")
    ]

    facts = []
    for verb_pos in verb_positions:
        pred = tagged_content_tokens[verb_pos][0]
        # if pred == "is":
        #     pred = "has_attribute"

        sbj = tagged_content_tokens[verb_pos - 1][0]
        try:
            obj = tagged_content_tokens[verb_pos + 1][0]
            # facts.append(elem_delim.join((sbj, pred, obj)))
            triplet_elements = [
                GraphFormat.HEAD_TOKEN,
                sbj,
                GraphFormat.TYPE_TOKEN,
                pred,
                GraphFormat.TAIL_TOKEN,
                obj,
            ]
            facts.append(" ".join(triplet_elements))
        except IndexError:
            continue

    for adj_pos in adj_positions:
        pred = "has_attribute"
        obj = tagged_content_tokens[adj_pos][0]
        sbj_candidates = [
            e[0]
            for i, e in enumerate(tagged_content_tokens)
            if i > adj_pos and e[1].startswith("N")
        ]
        try:
            sbj = (
                sbj_candidates[0]
                if sbj_candidates
                else tagged_content_tokens[adj_pos + 1][0]
            )
            # facts.append(elem_delim.join((sbj, pred, obj)))
            triplet_elements = [
                GraphFormat.HEAD_TOKEN,
                sbj,
                GraphFormat.TYPE_TOKEN,
                pred,
                GraphFormat.TAIL_TOKEN,
                obj,
            ]
            facts.append(" ".join(triplet_elements))
        except IndexError:
            continue

    if facts:
        # return fact_delim.join(facts)
        return " ".join(facts)
    else:
        # drop is fallback when rules don't find facts in text
        return blank_word(seq)


def rule(seq, is_graph):
    if is_graph:
        return graph2text(seq), False
    else:
        return text2graph(seq), True


def repeat_fact(seq, repeat_prob=0.2):
    # facts = seq.split(fact_delim)

    facts = GraphFormat.split_triples(seq)
    n = len(facts)

    for _ in range(n):
        if random.random() > repeat_prob:
            continue

        fact = random.choice(facts)
        pos = random.randint(0, len(facts))
        facts.insert(pos, fact)

    # return fact_delim.join(facts)
    return " ".join(facts)


def repeat_text(seq, repeat_prob=0.1):
    tokens = seq.split()
    new_tokens = []
    for t in tokens:
        new_tokens.append(t)
        if random.random() <= repeat_prob:
            new_tokens.append(t)
    return " ".join(new_tokens)


def repeat(seq, is_graph):
    if is_graph:
        return repeat_fact(seq), is_graph
    else:
        return repeat_text(seq), is_graph


existing_noise_functions = {
    "swap": swap,
    "drop": drop,
    "blank": blank,
    "rule": rule,
    "repeat": repeat,
}
