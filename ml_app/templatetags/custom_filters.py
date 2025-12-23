from django.template import Library

register = Library()

@register.filter
def lookup(dictionary, key):
    """
    Template filter to access dictionary values by key
    Usage: {{ dict|lookup:'key' }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''
