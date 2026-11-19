import pandas as pd

from corpus_converter import convert_corpus


def _row(source, sen_id, sen_num, token, categ):
    return {"source": source, "sen_id": sen_id, "sen_num": sen_num, "token": token, "categ": categ}


def test_convert_corpus_maps_collapsed_categories_to_target_labels():
    df = pd.DataFrame([
        _row("berlin", "p1", 1, "Ich", "D"),
        _row("berlin", "p1", 1, "mergen", "E"),
        _row("berlin", "p1", 1, ".", "O"),
        _row("berlin", "p1", 1, "Wiki", "SE"),
        _row("berlin", "p1", 1, "EStG", "SD"),
    ])

    records, exclusion_counts = convert_corpus(df)

    assert len(records) == 1
    record = records[0]
    assert record["tokens"] == ["Ich", "mergen", ".", "Wiki", "EStG"]
    assert record["labels"] == ["DE", "EN", "NEUT", "EN", "DE"]
    assert record["comment_id"] == "berlin:p1"
    assert exclusion_counts == {}


def test_convert_corpus_excludes_ambiguous_categories_and_reports_counts():
    df = pd.DataFrame([
        _row("berlin", "p1", 1, "Ich", "D"),
        _row("berlin", "p1", 1, "whatever", "M"),
        _row("berlin", "p1", 1, "thing", "SO"),
    ])

    records, exclusion_counts = convert_corpus(df)

    assert records[0]["tokens"] == ["Ich"]
    assert records[0]["labels"] == ["DE"]
    assert exclusion_counts == {"M": 1, "SO": 1}


def test_convert_corpus_drops_empty_tokens_and_groups_sentences_separately():
    df = pd.DataFrame([
        _row("berlin", "p1", 1, "Hallo", "D"),
        _row("berlin", "p1", 2, "Bye", "E"),
        _row("berlin", "p1", 1, "", "O"),  # blank EOS/EOP marker, must be dropped
    ])

    records, _ = convert_corpus(df)

    assert len(records) == 2
    all_tokens = {tuple(r["tokens"]) for r in records}
    assert ("Hallo",) in all_tokens
    assert ("Bye",) in all_tokens


def test_convert_corpus_drops_sentences_that_become_fully_ambiguous():
    df = pd.DataFrame([_row("berlin", "p1", 1, "blended", "M")])
    records, exclusion_counts = convert_corpus(df)
    assert records == []
    assert exclusion_counts == {"M": 1}