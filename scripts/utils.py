import re
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_lg") # Better at sentence boundaries but can be changed
user_pattern = re.compile(r"u/\w+")
subreddit_pattern = re.compile(r"r/\w+")

def check_root_verb(doc):
    """
    Checks for a complete sentence in a document, that is, a sentence with a verb (finite form) as the root

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A spaCy Doc object containing one or more sentences.

    Returns
    bool
        True if at least one sentence has a ROOT token that's a verb or auxiliary, otherwise False
    """
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                if "VerbForm=Fin" in token.morph:
                    return True
    return False


def is_valid_claim(text):
    """
    Determine whether a text qualifies as a valid claim based on
    syntactic and heuristic filtering rules.

    Parameters
    ----------
    text : str
        Input text to evaluate.

    Returns
    -------
    bool
        True if the text satisfies all validity criteria, otherwise False.

    Notes
    -----
    The function applies the following checks:

    - Contains at least one sentence with a finite root verb
      (via `check_root_verb`).
    - Does not contain URLs.
    - Does not contain the words "scam" or "legit".
    - Does not contain user mentions or subreddit mentions
      (based on predefined regex patterns).
    - Contains at least three tokens.
    - Contains no more than one exclamation mark and consists
      of a single sentence.
    - Has more than 60% alphabetic tokens.
    """
    doc = nlp(text)

    has_verb = check_root_verb(doc)
    no_url = not(any(token.like_url for token in doc))
    no_scam = not(any(token.text in ['scam', 'legit'] for token in doc))
    no_mentions = not(user_pattern.search(text) or subreddit_pattern.search(text))
    no_short_claims = len(doc) >= 3
    not_more_than_one_intj = text.count("!") <= 1 and len(list(doc.sents)) == 1 #typically like news ads or social media-esque posts, not claims


    alpha_tokens = [t for t in doc if t.is_alpha]
    frac_alpha = len(alpha_tokens) / max(1, len(doc))
    mostly_alpha = frac_alpha > 0.6

    return all([has_verb, no_url, no_scam, no_mentions, no_short_claims, not_more_than_one_intj, mostly_alpha])

def detect_comparison(text):
    """
    Detect whether a text contains explicit or implicit comparisons
    using rule-based spaCy pattern matching.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    tuple of (bool, str)
        A tuple containing:
        - bool: True if a comparison is detected, otherwise False.
        - str: The type of comparison detected:
            "STRICT_COMPARISON",
            "IMPLICIT_COMPARISON",
            or an empty string if none found.

    Notes
    -----
    Two categories of comparison patterns are used:

    STRICT_COMPARISON
        Explicit comparative constructions such as:
        - "as ADJ as"
        - "better than"
        - "more/less than"
        - "compared to"

    IMPLICIT_COMPARISON
        Comparative or superlative morphology without an explicit
        comparison target (e.g., adjectives tagged JJR or JJS,
        excluding cases like "most people").

    Strict comparisons are prioritized over implicit ones.
    """
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Strict comparison: explicitly compares two things/groups
    strict_patterns = [
        # as ADJ as , e.g as (brilliantly) great as#
        [{"LOWER": "as"}, {"POS": "ADV", "OP": "*"}, {"POS": "ADJ"}, {"LOWER": "as"}],
        # JJR than, e.g better than #
        [{"TAG": "JJR"}, {"LOWER": "than"}],
        # RBR ADJ/ADV than, e.g more determined than #
        [{"TAG": "RBR"}, {"TAG": "RB", "OP": "*"}, {"TAG": {"IN": ["JJ", "RB"]}}, {"LOWER": "than"}],
        # more/less ADJ than, e.g more pain than #
        [{"LOWER": {"IN": ["more", "less"]}}, {"POS": "NOUN"}, {"LOWER": "than"}],
        #  more than #
        [{"LOWER": {"IN": ["more", "less"]}}, {"LOWER": "than"}],
        # compare ADP, e.g compared to #
        [{"LEMMA": "compare", "TAG": "VBN"}, {"POS": "ADP"}],
        
    ]
    implicit_patterns = [
    # [{"TAG": {"IN": ["JJR", "JJS"]}}, {"POS": {"NOT_IN": ["NOUN"]}}],
    [{"TAG": {"IN": ["JJR", "JJS"]}, "LOWER": {"NOT_IN": ["most", "least"]}}], # Avoiding "most people"
    [{"TAG": "JJR"}, {"POS": "NOUN", "OP": "?"}],
    [{"LOWER": {"IN": ["more", "less"]}},{"POS": "NOUN"}],
    [{"LOWER": {"IN": ["more", "less"]}},{"POS": "ADV", "OP": "*"},{"POS": "ADJ"}],
    [{"TAG": "RBS"},{"POS": "ADV", "OP": "*"},{"POS": "ADJ"}],
    ]
    matcher.add("STRICT_COMPARISON", strict_patterns)
    matcher.add("IMPLICIT_COMPARISON", implicit_patterns)
    for sent in doc.sents:
        matches = matcher(sent)
        matches = [(doc.vocab.strings[match_id], start, end) for match_id, start, end in matches]

        if not matches:
            continue

        # Separate strict and implicit matches
        strict_matches = [m for m in matches if m[0] == "STRICT_COMPARISON"]
        implicit_matches = [m for m in matches if m[0] == "IMPLICIT_COMPARISON"]

        # Prioritize strict matches
        if strict_matches:
            return True, "STRICT_COMPARISON"
        elif implicit_matches:
            return True, "IMPLICIT_COMPARISON"
        # Code to not prioritize strict matches
        # for match_id, _, _ in matches:
        #     name = nlp.vocab.strings[match_id]
        #     return True, name         
    return False, ''

def detect_other(text): # Placeholder (think of 3-5 more nuanced categories)
    """
    Returns (mask: bool, type: str). Example placeholder.
    """
    
    return False,""


# --------------------------
# Category function mapping
# --------------------------
CATEGORY_FUNCTIONS = {
    "comparison": detect_comparison,
    "other": detect_other,
    # Add more categories as needed
}