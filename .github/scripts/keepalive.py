"""Keep the Streamlit Community Cloud demo awake.

Free-tier apps go to sleep after 12 hours without traffic, and a plain HTTP
request does not count as a visit. This opens the app in a headless browser,
clicks the wake-up button if the app is asleep, and waits until the app itself
has rendered. Exits with an error if it never does, so the workflow run fails.

Usage: python .github/scripts/keepalive.py APP_URL
"""

import re
import sys

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
APP_READY = '[data-testid="stAppViewContainer"]'
PAGE_TIMEOUT_MS = 60_000
WAKE_TIMEOUT_MS = 180_000  # a cold start can take a couple of minutes
ATTEMPTS = 2


def app_rendered(page) -> bool:
    """The app runs inside an iframe; look for Streamlit's main container in any frame."""
    for frame in page.frames:
        try:
            if frame.locator(APP_READY).count() > 0:
                return True
        except PlaywrightError:
            pass  # frame detached while checking
    return False


def wait_until_rendered(page, timeout_ms: int) -> bool:
    waited = 0
    while waited < timeout_ms:
        if app_rendered(page):
            return True
        page.wait_for_timeout(2_000)
        waited += 2_000
    return False


def visit(browser, url: str) -> str:
    """Return 'awake' or 'woken'. Raise RuntimeError if the app never renders."""
    page = browser.new_page()
    try:
        page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")

        button = page.get_by_role("button", name=WAKE_BUTTON)
        status = "awake"
        # Give the page time to show either the app or the sleep screen
        for _ in range(15):
            if app_rendered(page):
                break
            if button.count() > 0:
                button.first.click()
                status = "woken"
                break
            page.wait_for_timeout(2_000)

        if not wait_until_rendered(page, WAKE_TIMEOUT_MS):
            raise RuntimeError("app did not render in time")
        return status
    finally:
        page.close()


def main(url: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    print(f"{visit(browser, url)}: {url}")
                    return 0
                except (PlaywrightError, RuntimeError) as exc:
                    print(f"attempt {attempt} failed: {str(exc).splitlines()[0]}")
        finally:
            browser.close()
    print(f"FAILED: {url}")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
