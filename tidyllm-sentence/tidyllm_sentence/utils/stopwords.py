"""
English stop words list - most common words that carry little semantic meaning.
These are typically filtered out during text preprocessing for better embeddings.
"""

# Top 150+ English stop words
ENGLISH_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
    'was', 'will', 'with', 'would', 'you', 'your', 'i', 'me', 'my', 'we',
    'our', 'ours', 'they', 'them', 'their', 'theirs', 'she', 'her', 'hers',
    'him', 'his', 'this', 'these', 'that', 'those', 'am', 'is', 'are', 'was',
    'were', 'being', 'been', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'can', 'could', 'should', 'would', 'may', 'might', 'must',
    'shall', 'will', 'about', 'above', 'across', 'after', 'against', 'along',
    'among', 'around', 'because', 'before', 'behind', 'below', 'beneath',
    'beside', 'between', 'beyond', 'but', 'during', 'except', 'into', 'like',
    'near', 'over', 'since', 'through', 'throughout', 'till', 'under', 'until',
    'up', 'upon', 'within', 'without', 'all', 'another', 'any', 'anybody',
    'anyone', 'anything', 'both', 'each', 'either', 'everybody', 'everyone',
    'everything', 'few', 'many', 'neither', 'nobody', 'none', 'nothing',
    'one', 'other', 'others', 'several', 'some', 'somebody', 'someone',
    'something', 'most', 'much', 'no', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'now', 'here', 'there', 'when', 'where',
    'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'if', 'or', 'nor',
    'also', 'then', 'well', 'such', 'more', 'less', 'get', 'got', 'come',
    'came', 'go', 'went', 'see', 'saw', 'make', 'made', 'take', 'took',
    'give', 'gave', 'say', 'said', 'know', 'knew', 'think', 'thought',
    'back', 'out', 'off', 'away', 'down', 'still', 'even', 'quite',
    'rather', 'really', 'never', 'always', 'often', 'sometimes', 'usually'
}

# Minimal core set (most critical 25)
CORE_STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it',
    'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this',
    'but', 'his', 'by', 'from'
}

# Extended set for more aggressive filtering
EXTENDED_STOP_WORDS = ENGLISH_STOP_WORDS | {
    # Numbers as words
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
    'thousand', 'million', 'billion', 'first', 'second', 'third', 'fourth',
    'fifth', 'last', 'next', 'previous',
    
    # Web/digital artifacts
    'www', 'http', 'https', 'com', 'org', 'net', 'edu', 'gov', 'html',
    'pdf', 'doc', 'txt', 'email', 'mail', 'web', 'site', 'page', 'link',
    
    # Common interjections
    'oh', 'ah', 'eh', 'um', 'uh', 'yeah', 'yes', 'no', 'ok', 'okay',
    'hello', 'hi', 'hey', 'bye', 'goodbye', 'thanks', 'thank', 'please'
}

__all__ = ['ENGLISH_STOP_WORDS', 'CORE_STOP_WORDS', 'EXTENDED_STOP_WORDS', 
           'is_stopword', 'remove_stopwords']

def is_stopword(word: str, stop_set: set = None) -> bool:
    """Check if a word is a stop word."""
    if stop_set is None:
        stop_set = ENGLISH_STOP_WORDS
    return word.lower() in stop_set

def remove_stopwords(tokens: list, stop_set: set = None) -> list:
    """Remove stop words from a list of tokens."""
    if stop_set is None:
        stop_set = ENGLISH_STOP_WORDS
    return [token for token in tokens if not is_stopword(token, stop_set)]