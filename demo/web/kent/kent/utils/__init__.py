import re
import itertools

def expand_entities(text, entities):
    def next(text):
        for match in re.finditer(r'\{(.+?)(\?)?\}', text):
            if match.group(1) in entities:
                return match
        return None
    
    def do_expand(text):
        match = next(text)
        if not match:
            return [text]
        
        items = list(entities[match.group(1)])
        if match.group(2):
            items.append('')
            
        return [
            '{left}{item}{right}'.format(
                left=text[:match.start()],
                item=item,
                right=text[match.end():]
            )
            for item in items
        ]
    
    changed = True
    intermediates = [text]
    
    while changed:
        changed = False
        
        expanded = []
        for intermediate in intermediates:
            expanded.extend(do_expand(intermediate))
        
        if expanded != intermediates:
            changed = True

        intermediates = expanded
    
    return [
        re.sub(r'\s{2,}', ' ', expanded_i).strip()
        for expanded_i in expanded
    ]