import re
import urllib
from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    # Add PDF detection to URL validation
    decoded_url = urllib.parse.unquote(url).lower()
    if '.pdf' in decoded_url:
        return False
    
    # Enhanced fake URL detection
    fake_patterns = [
        r'example\.(com|org|net)',
        r'\b(test|sample|dummy|placeholder)\.',
        r'\b(mock|fake|staging|dev)\.(com|org|net)\b'
    ]
    if any(re.search(p, url, re.IGNORECASE) for p in fake_patterns):
        return False
    
    if not url:
        return False

    # Check for example/fake URLs
    fake_url_patterns = [
        r'example\.com',
        r'sample\.org',
        r'test\.net',
        r'domain\.com',
        r'yourcompany\.com',
        r'acme\.com',
        r'widget\.com',
        r'placeholder\.net',
        r'company\.org'
    ]

    for pattern in fake_url_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return False

    # Basic URL validation
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False