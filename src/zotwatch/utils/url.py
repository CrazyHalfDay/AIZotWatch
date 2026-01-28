"""URL validation utilities for SSRF prevention."""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Allowed URL schemes
ALLOWED_SCHEMES = frozenset({"http", "https"})

# Known safe domains for academic content
SAFE_DOMAINS = frozenset({
    "doi.org",
    "dx.doi.org",
    "arxiv.org",
    "export.arxiv.org",
    "api.crossref.org",
    "api.eartharxiv.org",
    "api.zotero.org",
    # Major publishers
    "acm.org",
    "dl.acm.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "link.springer.com",
    "elsevier.com",
    "sciencedirect.com",
    "wiley.com",
    "onlinelibrary.wiley.com",
    "nature.com",
    "science.org",
    "pnas.org",
    "agu.org",
    "agupubs.onlinelibrary.wiley.com",
    "tandfonline.com",
    "mdpi.com",
    "frontiersin.org",
    "plos.org",
    "journals.plos.org",
    "oup.com",
    "academic.oup.com",
    "cambridge.org",
    "sagepub.com",
    "journals.sagepub.com",
    "biomedcentral.com",
    "bmc.com",
    "cell.com",
    "jstor.org",
    "researchgate.net",
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
})


class URLValidationError(ValueError):
    """Raised when URL validation fails."""

    pass


def is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/internal IP address.

    Args:
        hostname: Hostname to check.

    Returns:
        True if hostname resolves to private IP, False otherwise.
    """
    try:
        # Resolve hostname to IP
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)

        # Check if IP is private, loopback, or link-local
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except (socket.gaierror, ValueError):
        # If resolution fails, treat as potentially unsafe
        return True


def validate_url(
    url: str,
    *,
    allow_private_ip: bool = False,
    require_safe_domain: bool = False,
) -> str:
    """Validate URL for safe fetching (SSRF prevention).

    Args:
        url: URL to validate.
        allow_private_ip: Whether to allow private/internal IPs.
        require_safe_domain: Whether to require domain in SAFE_DOMAINS list.

    Returns:
        The validated URL (normalized).

    Raises:
        URLValidationError: If URL is invalid or unsafe.
    """
    if not url:
        raise URLValidationError("Empty URL")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise URLValidationError(f"Invalid URL format: {e}") from e

    # Check scheme
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLValidationError(
            f"Invalid URL scheme '{scheme}'. Allowed: {', '.join(ALLOWED_SCHEMES)}"
        )

    # Check hostname exists
    hostname = parsed.hostname
    if not hostname:
        raise URLValidationError("URL missing hostname")

    # Check for private IPs
    if not allow_private_ip and is_private_ip(hostname):
        raise URLValidationError(f"URL resolves to private/internal IP: {hostname}")

    # Check against safe domain list if required
    if require_safe_domain:
        # Check if hostname matches or is subdomain of safe domain
        hostname_lower = hostname.lower()
        is_safe = False
        for safe_domain in SAFE_DOMAINS:
            if hostname_lower == safe_domain or hostname_lower.endswith(f".{safe_domain}"):
                is_safe = True
                break
        if not is_safe:
            raise URLValidationError(f"Domain not in safe list: {hostname}")

    return url


def is_safe_url(
    url: str,
    *,
    allow_private_ip: bool = False,
    require_safe_domain: bool = False,
) -> bool:
    """Check if URL is safe for fetching.

    Args:
        url: URL to check.
        allow_private_ip: Whether to allow private/internal IPs.
        require_safe_domain: Whether to require domain in SAFE_DOMAINS list.

    Returns:
        True if URL is safe, False otherwise.
    """
    try:
        validate_url(
            url,
            allow_private_ip=allow_private_ip,
            require_safe_domain=require_safe_domain,
        )
        return True
    except URLValidationError:
        return False


__all__ = [
    "URLValidationError",
    "validate_url",
    "is_safe_url",
    "is_private_ip",
    "ALLOWED_SCHEMES",
    "SAFE_DOMAINS",
]
