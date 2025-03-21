import re
from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted and not a fabricated example URL.
    """
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