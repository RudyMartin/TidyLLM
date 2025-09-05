"""
Pure Python stemmer implementation - no NLTK required.
Based on simplified Porter Stemmer rules for educational transparency.
"""

__all__ = ['simple_stem', 'porter_stem']

def simple_stem(word: str) -> str:
    """Very simple suffix stripping for common English patterns."""
    if isinstance(word, list):
        raise TypeError(f"simple_stem() expects a string, got {type(word)}. Use [simple_stem(w) for w in word_list] for lists.")
    
    word = str(word).lower()
    
    # Length check
    if len(word) <= 3:
        return word
    
    # Common suffix removal (order matters!)
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'  # cities -> city
    
    if word.endswith('sses'):
        return word[:-2]  # classes -> class
        
    if word.endswith('ied'):
        return word[:-3] + 'y' if len(word) > 4 else word[:-2]
        
    if word.endswith('ing'):
        stem = word[:-3]
        if len(stem) > 2:
            # Handle doubling: running -> run
            if len(stem) > 1 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
                return stem[:-1]
            return stem
            
    if word.endswith('ed'):
        stem = word[:-2]
        if len(stem) > 2:
            # Handle doubling: stopped -> stop
            if len(stem) > 1 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
                return stem[:-1]
            return stem
            
    if word.endswith('er'):
        stem = word[:-2]
        if len(stem) > 2:
            return stem
            
    if word.endswith('est'):
        stem = word[:-3]
        if len(stem) > 2:
            return stem
            
    if word.endswith('ly'):
        stem = word[:-2]
        if len(stem) > 2:
            return stem
            
    if word.endswith('ness'):
        stem = word[:-4]
        if len(stem) > 2:
            return stem
            
    if word.endswith('ment'):
        stem = word[:-4]
        if len(stem) > 2:
            return stem
            
    if word.endswith('tion'):
        stem = word[:-4]
        if len(stem) > 2:
            return stem + 'te' if stem.endswith('a') else stem
            
    if word.endswith('s') and not word.endswith('ss'):
        stem = word[:-1]
        if len(stem) > 2:
            return stem
    
    return word


def porter_stem(word: str) -> str:
    """
    Simplified Porter Stemmer implementation.
    Educational version with main rules only.
    """
    word = word.lower()
    
    # Step 1a: Plurals
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif word.endswith('ss'):
        pass  # Do nothing
    elif word.endswith('s'):
        word = word[:-1]
    
    # Step 1b: Past tense and progressive
    if word.endswith('eed'):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith('ed'):
        stem = word[:-2]
        if _contains_vowel(stem):
            word = stem
            word = _step1b_cleanup(word)
    elif word.endswith('ing'):
        stem = word[:-3]
        if _contains_vowel(stem):
            word = stem
            word = _step1b_cleanup(word)
    
    # Step 1c: Y -> I
    if word.endswith('y'):
        stem = word[:-1]
        if _contains_vowel(stem):
            word = stem + 'i'
    
    # Step 2: Double suffixes
    word = _step2(word)
    
    # Step 3: More suffixes
    word = _step3(word)
    
    # Step 4: Even more suffixes
    word = _step4(word)
    
    # Step 5a: E removal
    if word.endswith('e'):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    
    # Step 5b: Double consonant removal
    if len(word) > 1 and word[-1] == word[-2] and word[-1] == 'l':
        if _measure(word[:-1]) > 1:
            word = word[:-1]
    
    return word


def _contains_vowel(word: str) -> bool:
    """Check if word contains a vowel."""
    return any(c in 'aeiou' for c in word)


def _measure(word: str) -> int:
    """Count VC sequences (simplified)."""
    cv_pattern = []
    for c in word:
        if c in 'aeiou':
            cv_pattern.append('V')
        else:
            cv_pattern.append('C')
    
    # Simplify consecutive letters
    simplified = []
    for i, c in enumerate(cv_pattern):
        if i == 0 or c != cv_pattern[i-1]:
            simplified.append(c)
    
    # Count VC pairs
    vc_count = 0
    for i in range(len(simplified) - 1):
        if simplified[i] == 'V' and simplified[i+1] == 'C':
            vc_count += 1
    
    return vc_count


def _ends_cvc(word: str) -> bool:
    """Check if word ends with CVC pattern where last C is not w, x, or y."""
    if len(word) < 3:
        return False
    
    c1 = word[-3] not in 'aeiou'
    v = word[-2] in 'aeiou'
    c2 = word[-1] not in 'aeiouxwy'
    
    return c1 and v and c2


def _step1b_cleanup(word: str) -> str:
    """Cleanup after step 1b."""
    if word.endswith(('at', 'bl', 'iz')):
        return word + 'e'
    elif len(word) > 1 and word[-1] == word[-2] and word[-1] not in 'lsz':
        return word[:-1]
    elif _measure(word) == 1 and _ends_cvc(word):
        return word + 'e'
    return word


def _step2(word: str) -> str:
    """Step 2 of Porter Stemmer."""
    suffixes = {
        'ational': 'ate',
        'tional': 'tion',
        'enci': 'ence',
        'anci': 'ance',
        'izer': 'ize',
        'abli': 'able',
        'alli': 'al',
        'entli': 'ent',
        'eli': 'e',
        'ousli': 'ous',
        'ization': 'ize',
        'ation': 'ate',
        'ator': 'ate',
        'alism': 'al',
        'iveness': 'ive',
        'fulness': 'ful',
        'ousness': 'ous',
        'aliti': 'al',
        'iviti': 'ive',
        'biliti': 'ble'
    }
    
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement
    return word


def _step3(word: str) -> str:
    """Step 3 of Porter Stemmer."""
    suffixes = {
        'icate': 'ic',
        'ative': '',
        'alize': 'al',
        'iciti': 'ic',
        'ical': 'ic',
        'ful': '',
        'ness': ''
    }
    
    for suffix, replacement in suffixes.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement
    return word


def _step4(word: str) -> str:
    """Step 4 of Porter Stemmer."""
    suffixes = ['al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant',
                'ement', 'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti',
                'ous', 'ive', 'ize']
    
    for suffix in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 1:
                if suffix == 'ion':
                    if len(stem) > 0 and stem[-1] in 'st':
                        return stem
                else:
                    return stem
    return word