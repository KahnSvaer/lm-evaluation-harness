from functools import partial

import datasets


def process_docs(dataset: datasets.Dataset, language: str) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "language": doc["language"],
            "context": doc["document_title"] + "\n" + doc["document_plaintext"],
            "question": doc["question_text"],
            "answers": doc["annotations"]["passage_answer_candidate_index"],
        }
        return out_doc

    new_dataset = dataset.map(_process_doc)
    assert isinstance(new_dataset, datasets.Dataset), type(new_dataset)
    return new_dataset.filter(lambda x: x["language"] == language)


def gold_has_passage_answer(gold_label_list, passage_non_null_threshold):
    """Gets vote from annotators for judging if there is a passage answer."""

    gold_has_answer = gold_label_list and (
        sum(
            [
                label.passage_answer_index >= 0  # passage answer not null
                for label in gold_label_list  # for each annotator
            ]
        )
        >= passage_non_null_threshold
    )

    return gold_has_answer


def process_results(doc, results):
    pass


process_arabic = partial(process_docs, language="arabic")
process_bengali = partial(process_docs, language="bengali")
process_english = partial(process_docs, language="english")
process_finnish = partial(process_docs, language="finnish")
process_indonesian = partial(process_docs, language="indonesian")
process_japanese = partial(process_docs, language="japanese")
process_korean = partial(process_docs, language="korean")
process_russian = partial(process_docs, language="russian")
process_swahili = partial(process_docs, language="swahili")
process_telugu = partial(process_docs, language="telugu")
process_thai = partial(process_docs, language="thai")
