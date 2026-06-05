"""Anti-detect browser with Cloudflare Turnstile bypass using Camoufox.

Features:
- Camoufox (Firefox-based anti-detect browser)
- Turnstile bypass via checkbox click (using camoufox-captcha)
- Retry mechanism with exponential backoff
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Cloudflare challenge detection patterns
# Note: "challenge-platform" is too generic - it can appear in regular pages
# Focus on title/visible text indicators that only appear on challenge pages
CLOUDFLARE_TITLE_INDICATORS = [
    "Just a moment...",
    "Checking your browser",
]

CLOUDFLARE_BODY_INDICATORS = [
    "Verify you are human",
    "Please wait while we verify your browser",
    "Enable JavaScript and cookies to continue",
]

# Path for persistent browser profile
DEFAULT_PROFILE_PATH = Path.home() / ".cache" / "zotwatch" / "camoufox_profile"

# Resource types to abort when resource blocking is enabled. We only need the
# document/scripts/XHR to obtain the abstract, so images/fonts/styles/media are
# pure overhead and slow down page loads.
BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}

# Smart-settle polling: instead of always waiting for networkidle, poll the DOM
# and return as soon as the abstract is extractable.
SETTLE_POLL_INTERVAL_S = 0.5
SETTLE_MAX_WAIT_S = 15.0

# An extraction readiness probe: (html, final_url) -> bool
ReadyCheck = Callable[[str, "str | None"], bool]


class StealthBrowser:
    """Anti-detect browser with Cloudflare Turnstile bypass using Camoufox.

    Uses Camoufox (Firefox-based anti-detect browser) with:
    - Turnstile checkbox click bypass (using camoufox-captcha)

    Thread-safe: supports concurrent page fetching.
    """

    _browser = None
    _context = None
    _camoufox_ctx = None
    _initialized = False
    _init_lock = threading.Lock()
    _profile_path = DEFAULT_PROFILE_PATH
    _event_loop = None
    _loop_thread = None

    # Configuration
    DEFAULT_TIMEOUT = 60000
    MAX_CF_RETRIES = 3

    @classmethod
    def set_profile_path(cls, path: Path) -> None:
        """Override default profile path (must be called before get_browser)."""
        with cls._init_lock:
            if cls._initialized:
                logger.warning("StealthBrowser already initialized; profile path change ignored")
                return
            cls._profile_path = Path(path)

    @classmethod
    def _ensure_event_loop(cls):
        """Ensure we have an event loop running in a background thread."""
        if cls._event_loop is None or not cls._event_loop.is_running():
            cls._event_loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(cls._event_loop)
                cls._event_loop.run_forever()

            cls._loop_thread = threading.Thread(target=run_loop, daemon=True)
            cls._loop_thread.start()
            # Wait for loop to start
            time.sleep(0.1)

    @classmethod
    def _run_async(cls, coro):
        """Run async code from sync context."""
        cls._ensure_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, cls._event_loop)
        return future.result(timeout=120)

    @classmethod
    def get_browser(cls):
        """Get or create Camoufox browser instance with persistent profile."""
        with cls._init_lock:
            if cls._initialized and cls._browser:
                return cls._browser, cls._context

            cls._initialized = True
            try:
                # Ensure profile directory exists
                cls._profile_path.mkdir(parents=True, exist_ok=True)

                # Initialize browser using async API
                cls._browser, cls._context = cls._run_async(cls._init_browser_async())
                logger.info("Camoufox browser initialized")
                return cls._browser, cls._context
            except Exception as e:
                logger.warning("Failed to initialize Camoufox browser: %s", e)
                cls._browser = None
                cls._context = None
                return None, None

    @classmethod
    async def _init_browser_async(cls):
        """Initialize Camoufox browser asynchronously."""
        from camoufox import AsyncCamoufox

        # Create browser with anti-detect settings
        # Note: persistent_context causes issues with Cloudflare bypass
        # We'll manage cookies separately if needed
        # Keep a reference to the AsyncCamoufox context manager so we can
        # properly close Playwright and its background tasks on shutdown.
        cls._camoufox_ctx = AsyncCamoufox(
            headless=True,
            geoip=True,
            # Required for camoufox-captcha to traverse Shadow DOM
            config={"forceScopeAccess": True},
            # Required for camoufox-captcha (suppress warning)
            disable_coop=True,
            i_know_what_im_doing=True,
            # Enable human-like mouse movements and interactions
            humanize=True,
        )

        browser = await cls._camoufox_ctx.__aenter__()

        # Browser acts as context for new_page
        return browser, browser

    @classmethod
    def _is_cloudflare_challenge(cls, html: str) -> bool:
        """Detect if page contains Cloudflare challenge.

        Uses a combination of title and body indicators to avoid
        false positives (e.g., "challenge-platform" appearing in regular pages).
        """
        if not html:
            return False
        html_lower = html.lower()

        # Check for title indicators (most reliable)
        for indicator in CLOUDFLARE_TITLE_INDICATORS:
            if indicator.lower() in html_lower:
                return True

        # Check for body indicators
        for indicator in CLOUDFLARE_BODY_INDICATORS:
            if indicator.lower() in html_lower:
                return True

        return False

    @classmethod
    async def _solve_cloudflare_interstitial(cls, page) -> bool:
        """Solve Cloudflare interstitial (full-page) challenge.

        Args:
            page: Camoufox page object.

        Returns:
            True if solved, False otherwise.
        """
        try:
            from camoufox_captcha import solve_captcha

            logger.info("Attempting to solve Cloudflare interstitial challenge...")

            # For interstitial (full-page) challenges, pass the page directly
            success = await solve_captcha(
                page,
                captcha_type="cloudflare",
                challenge_type="interstitial",
                solve_attempts=3,
                solve_click_delay=3.0,  # Longer delay after click
            )

            if success:
                logger.info("Cloudflare interstitial challenge clicked!")

                # Wait longer for page navigation/redirect
                await asyncio.sleep(5)

                try:
                    await page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass

                # Additional wait for page to settle
                await asyncio.sleep(3)

                # Check if we passed the challenge
                html = await cls._safe_content(page)
                if not cls._is_cloudflare_challenge(html):
                    logger.info("Cloudflare bypass confirmed!")
                    return True
                else:
                    logger.warning("Still on challenge page after click, waiting more...")
                    # Wait more and check again
                    await asyncio.sleep(5)
                    html = await cls._safe_content(page)
                    return not cls._is_cloudflare_challenge(html)
            else:
                logger.warning("Cloudflare interstitial solve failed")
                return False

        except ImportError:
            logger.warning("camoufox-captcha not installed")
            return False
        except Exception as e:
            logger.warning("Cloudflare interstitial solve failed: %s", e)
            return False

    @classmethod
    async def _solve_turnstile_widget(cls, page) -> bool:
        """Solve embedded Turnstile widget by clicking the checkbox.

        Args:
            page: Camoufox page object.

        Returns:
            True if solved, False otherwise.
        """
        try:
            from camoufox_captcha import solve_captcha

            logger.info("Attempting to solve Turnstile widget...")

            # Try to find Turnstile widget container
            selectors = [
                "div.cf-turnstile",
                "[data-turnstile-widget]",
                'iframe[src*="challenges.cloudflare.com"]',
            ]

            container = None
            for selector in selectors:
                try:
                    container = await page.wait_for_selector(selector, timeout=5000)
                    if container:
                        logger.debug("Found Turnstile widget: %s", selector)
                        break
                except Exception:
                    continue

            if not container:
                logger.debug("No Turnstile widget found")
                return False

            # Use camoufox-captcha to solve the widget
            success = await solve_captcha(
                container,
                captcha_type="cloudflare",
                challenge_type="turnstile",
                solve_attempts=3,
                solve_click_delay=2.0,
            )

            if success:
                logger.info("Turnstile widget solved!")
                await asyncio.sleep(3)
                return True
            else:
                logger.warning("Turnstile widget solve failed")
                return False

        except ImportError:
            logger.warning("camoufox-captcha not installed")
            return False
        except Exception as e:
            logger.warning("Turnstile widget solve failed: %s", e)
            return False

    @classmethod
    async def _solve_turnstile_manual_click(cls, page) -> bool:
        """Fallback: manually click the Turnstile checkbox by coordinates.

        Args:
            page: Camoufox page object.

        Returns:
            True if solved, False otherwise.
        """
        try:
            logger.info("Attempting manual Turnstile checkbox click...")

            # Find Cloudflare iframe
            for _ in range(15):
                await asyncio.sleep(1)

                frames = page.frames
                cf_frame = None
                for frame in frames:
                    if frame.url.startswith("https://challenges.cloudflare.com"):
                        cf_frame = frame
                        break

                if cf_frame:
                    break

            if not cf_frame:
                logger.warning("Could not find Cloudflare challenge iframe")
                return False

            # Get iframe element and its bounding box
            iframe_element = await page.query_selector('iframe[src*="challenges.cloudflare.com"]')
            if not iframe_element:
                logger.warning("Could not find iframe element")
                return False

            box = await iframe_element.bounding_box()
            if not box:
                logger.warning("Could not get iframe bounding box")
                return False

            # Calculate click coordinates (checkbox is typically in the left portion)
            click_x = box["x"] + box["width"] / 9
            click_y = box["y"] + box["height"] / 2

            logger.debug("Clicking at coordinates: (%.1f, %.1f)", click_x, click_y)

            # Click the checkbox
            await page.mouse.click(click_x, click_y)

            # Wait for verification
            await asyncio.sleep(5)

            # Check if challenge is resolved
            html = await cls._safe_content(page)
            if not cls._is_cloudflare_challenge(html):
                logger.info("Turnstile bypassed via manual click!")
                return True

            logger.warning("Manual click did not bypass challenge")
            return False

        except Exception as e:
            logger.warning("Manual Turnstile click failed: %s", e)
            return False

    @classmethod
    async def _handle_cloudflare_async(cls, page) -> bool:
        """Handle Cloudflare challenge on page.

        Args:
            page: Camoufox page object.

        Returns:
            True if challenge was bypassed, False otherwise.
        """
        html = await cls._safe_content(page)
        if not cls._is_cloudflare_challenge(html):
            return True

        logger.info("Cloudflare challenge detected, attempting bypass...")

        # Wait for challenge to fully render
        await asyncio.sleep(3)

        # Method 1: Try interstitial (full-page) challenge solver
        # This handles the common case of Cloudflare interstitial pages
        if await cls._solve_cloudflare_interstitial(page):
            return True

        # Re-check if still on challenge page
        html = await cls._safe_content(page)
        if not cls._is_cloudflare_challenge(html):
            return True

        # Method 2: Try embedded Turnstile widget solver
        if await cls._solve_turnstile_widget(page):
            html = await cls._safe_content(page)
            if not cls._is_cloudflare_challenge(html):
                return True

        # Method 3: Try manual coordinate-based click
        if await cls._solve_turnstile_manual_click(page):
            return True

        return False

    @classmethod
    async def _enable_resource_blocking(cls, page) -> None:
        """Abort image/font/stylesheet/media requests to speed up page loads."""

        async def _handler(route):
            try:
                if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
                    await route.abort()
                else:
                    await route.continue_()
            except Exception:
                # Never let a routing error break the fetch
                try:
                    await route.continue_()
                except Exception:
                    pass

        try:
            await page.route("**/*", _handler)
        except Exception as e:
            logger.debug("Failed to enable resource blocking: %s", e)

    @classmethod
    async def _safe_content(cls, page, retries: int = 8, delay: float = 0.4) -> str:
        """Return page HTML, tolerating transient "page is navigating" errors.

        ``page.content()`` raises while the page is mid-navigation, e.g. the
        ``linkinghub.elsevier.com`` -> ``sciencedirect.com`` client-side
        redirect. Rather than letting that abort the whole fetch, retry briefly
        until the navigation settles; return "" if it never does.
        """
        for _ in range(retries):
            try:
                return await page.content()
            except Exception as e:
                if "navigating" in str(e).lower():
                    await asyncio.sleep(delay)
                    continue
                raise
        return ""

    @classmethod
    async def _settle_page(
        cls,
        page,
        ready_check: ReadyCheck | None,
    ) -> tuple[str, str | None]:
        """Wait for the page to become extractable, returning as early as possible.

        With a ``ready_check``, poll the DOM and return the moment the abstract
        is extractable (or a Cloudflare challenge is detected). Without one, fall
        back to a single networkidle wait. This avoids always burning the full
        15s settle budget on pages that are ready immediately.
        """
        if ready_check is None:
            try:
                await page.wait_for_load_state("networkidle", timeout=int(SETTLE_MAX_WAIT_S * 1000))
            except Exception:
                pass
            return await cls._safe_content(page), page.url

        elapsed = 0.0
        html = await cls._safe_content(page)
        final_url = page.url
        while elapsed < SETTLE_MAX_WAIT_S:
            html = await cls._safe_content(page)
            final_url = page.url
            # Hand off to the Cloudflare handler as soon as a challenge appears
            if cls._is_cloudflare_challenge(html):
                return html, final_url
            try:
                if ready_check(html, final_url):
                    logger.debug("Page ready (abstract extractable) after %.1fs", elapsed)
                    return html, final_url
            except Exception as e:
                logger.debug("ready_check raised: %s", e)
            await asyncio.sleep(SETTLE_POLL_INTERVAL_S)
            elapsed += SETTLE_POLL_INTERVAL_S

        return html, final_url

    @classmethod
    async def _fetch_page_async(
        cls,
        browser,
        context,
        url: str,
        timeout: int,
        max_retries: int,
        ready_check: ReadyCheck | None = None,
        block_resources: bool = True,
    ) -> tuple[str | None, str | None]:
        """Async implementation of page fetching."""
        for attempt in range(max_retries):
            page = None
            try:
                page = await context.new_page()

                if block_resources:
                    await cls._enable_resource_blocking(page)

                logger.debug("Navigating to %s (attempt %d/%d)", url, attempt + 1, max_retries)

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                except Exception as e:
                    logger.debug("Navigation exception (may be normal): %s", str(e)[:100])

                # Smart settle: return as soon as the abstract is extractable
                html, final_url = await cls._settle_page(page, ready_check)

                # Handle Cloudflare if detected
                if cls._is_cloudflare_challenge(html):
                    success = await cls._handle_cloudflare_async(page)

                    if success:
                        # Wait a bit for page to fully load after bypass
                        await asyncio.sleep(2)
                        html = await cls._safe_content(page)
                        final_url = page.url

                        if not cls._is_cloudflare_challenge(html):
                            logger.info("Cloudflare bypassed successfully!")
                            return html, final_url

                    # Retry if bypass failed
                    if attempt < max_retries - 1:
                        logger.info(
                            "Cloudflare bypass attempt %d/%d failed, retrying...",
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(5)
                        continue

                    return html, final_url

                # No challenge detected
                return html, final_url

            except Exception as e:
                logger.warning(
                    "Fetch attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    repr(e),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)

            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass

        return None, None

    @classmethod
    def fetch_page(
        cls,
        url: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_CF_RETRIES,
        ready_check: ReadyCheck | None = None,
        block_resources: bool = True,
    ) -> tuple[str | None, str | None]:
        """Fetch page content with Cloudflare Turnstile bypass.

        Args:
            url: URL to fetch.
            timeout: Timeout in milliseconds.
            max_retries: Maximum retry attempts.
            ready_check: Optional probe ``(html, final_url) -> bool``; when it
                returns True the page is considered ready and returned early,
                instead of waiting for the full settle budget.
            block_resources: Abort image/font/stylesheet/media requests.

        Returns:
            Tuple of (html_content, final_url) or (None, None) on failure.
        """
        # Validate URL to prevent SSRF attacks
        from zotwatch.utils.url import URLValidationError, validate_url

        try:
            validate_url(url, allow_private_ip=False)
        except URLValidationError as e:
            logger.warning("URL validation failed for %s: %s", url, e)
            return None, None

        browser, context = cls.get_browser()
        if browser is None:
            return None, None

        try:
            return cls._run_async(
                cls._fetch_page_async(
                    browser,
                    context,
                    url,
                    timeout,
                    max_retries,
                    ready_check=ready_check,
                    block_resources=block_resources,
                )
            )
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, repr(e))
            return None, None

    @classmethod
    def clear_profile(cls) -> None:
        """Clear the persistent browser profile (cookies and data)."""
        import shutil

        try:
            if cls._profile_path.exists():
                shutil.rmtree(cls._profile_path)
                logger.info("Browser profile cleared: %s", cls._profile_path)
        except Exception as e:
            logger.warning("Failed to clear profile: %s", e)

    @classmethod
    def close(cls):
        """Clean up browser resources."""
        with cls._init_lock:
            # Close Camoufox/Playwright context first so that all background
            # tasks (including Playwright's Connection.run) are shut down
            # cleanly before we stop the event loop.
            if cls._camoufox_ctx:
                try:
                    cls._run_async(cls._camoufox_ctx.__aexit__(None, None, None))
                except Exception as e:
                    logger.debug("Error closing Camoufox context: %s", e)
                cls._camoufox_ctx = None

            cls._browser = None
            cls._context = None

            if cls._event_loop:
                try:
                    cls._event_loop.call_soon_threadsafe(cls._event_loop.stop)
                except Exception:
                    pass
                cls._event_loop = None
                cls._loop_thread = None

            cls._initialized = False


__all__ = ["StealthBrowser"]
