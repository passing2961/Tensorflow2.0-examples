import re

_josa_table = {
    '은': ('은', '는'),
    '는': ('은', '는'),
    '이': ('이', '가'),
    '가': ('이', '가'),
    '와': ('와', '과'),
    '과': ('와', '과'),
    '을': ('을', '를'),
    '를': ('을', '를'),
    '으로': ('으로', '로'),
    '로': ('으로', '로'),
}
_josa_pattern = re.compile(r'([가-힣])/({josa})\b'.format(
    josa=r'|'.join(map(re.escape, sorted(_josa_table.keys(), key=len, reverse=True)))
))

def _has_jongsung(syllable):
    return ((ord(syllable) - 0xAC00) % 28) > 0

def _choose_josa(syllable, josa):
    index = 0 if _has_jongsung(syllable) else 1
    return _josa_table[josa][index]

def resolve_josa(text):
    def replacement(match):
        syllable, josa = match.groups()
        return syllable + _choose_josa(syllable, josa)
    
    return _josa_pattern.sub(replacement, text)