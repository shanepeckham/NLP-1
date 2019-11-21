import base64
import json

from os.path import join as JP

from glob import glob

import re


def all_files_exist(doc_filepath):
    """
    Checks all required files are present in patent folder

    Args:
        doc_filepath: (str) path to a given patent folder

    Returns: True if all files present, False otherwise

    """
    docnorm_path = glob(JP(doc_filepath, 'oc_docnorm_*.json'))
    semannot_path = glob(JP(doc_filepath, 'oc_semantic_annotation_*.json'))
    return all([docnorm_path, semannot_path])


def get_patent_docs(doc_filepath):
    """
    Gets patent text from docnorm and semantic annotation files.
    It returns to versions of it: as parsed sentences and as parsed paragraphs

    Args:
        doc_filepath: (str) path to a given patent folder

    Returns: (tuple) Parsed sentences (as list), Parsed paragraphs (as list)

    """
    with open([p for p in glob(JP(doc_filepath, 'oc_docnorm_*.json'))][0], 'r') as fp:
        docnorm = json.load(fp)

    with open([p for p in glob(JP(doc_filepath, 'oc_semantic_annotation_*.json'))][0], 'r') as fp:
        annotations = json.load(fp)

    mparts = annotations['doc']['mparts']

    text_structure = [mp['semantic_enrichment']['text_structure'] for mp in mparts if 'text_structure' in mp['semantic_enrichment']][0]
    sentences = text_structure['sentences']
    paragraphs = text_structure['paragraphs']

    mentions = [mp['semantic_enrichment']['mentions'] for mp in mparts if 'mentions' in mp['semantic_enrichment']][0]

    docnorm_text = base64.b64decode(docnorm["text"]).decode('latin1').replace("\x00", "")

    parsed_sentences = [{"id": sentence["sentence_id"], "text": docnorm_text[sentence['start']:sentence['end']]} for sentence in sentences]
    parsed_paragraphs = [{"id": paragraph["paragraph_id"], "text": docnorm_text[paragraph['start']:paragraph['end']]} for paragraph in paragraphs]

    return parsed_sentences, parsed_paragraphs


def join_text_from_fragments(parsed_fragments, min_frag_len=30):
    """
    Joins all fragments of text from a pre parsed format (paragraphs,
    sentences, etc.) into a single string object. Fragment whose length
    is lower than a predefined length are ignored.

    Args:
        parsed_fragments: (list) containing parsed fragments of text
        min_frag_len: (int) minimum length a fragment must have to be considered

    Returns: (str) joined text

    """

    raw_doc = " ".join([par["text"] for par in parsed_fragments if len(par["text"]) > min_frag_len])

    return raw_doc


def process_text(doc):
    """
    Basic processing of raw text

    Args:
        doc: (str) raw document text

    Returns: (str) Processed text

    """

    # Remove line jumps
    doc = re.sub(r"\n", " ", doc)

    # Convert to lowercase
    doc = doc.lower()

    # Remove punctuations
    doc = re.sub('[^a-zA-Z]', ' ', doc)

    # remove special characters and digits
    doc = re.sub("(\\d|\\W)+", " ", doc)

    # Remove numbers between square brackets (typical notation in most patents)
    doc = re.sub(r'\[[0-9]+]', "", doc)

    # Remove extra spaces
    doc = re.sub(r" +", " ", doc)

    return doc

