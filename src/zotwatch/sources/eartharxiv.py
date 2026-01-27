"""EarthArXiv source implementation via OAI-PMH API.

EarthArXiv preprint server for Earth sciences, hosted by California Digital Library.
API documentation: https://eartharxiv.github.io/faq.html

The new EarthArXiv uses OAI-PMH protocol (not OSF API since 2021 migration).
OAI-PMH endpoint: https://eartharxiv.org/api/oai/
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

import requests

from zotwatch.config.settings import Settings
from zotwatch.core.constants import DEFAULT_HTTP_TIMEOUT
from zotwatch.core.exceptions import SourceFetchError
from zotwatch.core.models import CandidateWork
from zotwatch.utils.datetime import utc_yesterday_end

from .base import BaseSource, SourceRegistry, clean_title

logger = logging.getLogger(__name__)

# OAI-PMH namespaces
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

# EarthArXiv OAI-PMH endpoint (new CDL-hosted platform)
EARTHARXIV_OAI_URL = "https://eartharxiv.org/api/oai/"


@SourceRegistry.register
class EartharxivSource(BaseSource):
    """EarthArXiv preprint source via OAI-PMH protocol."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.config = settings.sources.eartharxiv
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ZotWatch/1.0 (https://github.com/CrazyHalfDay/AIZotWatch)",
            "Accept": "application/xml",
        })

    @property
    def name(self) -> str:
        return "eartharxiv"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def fetch(self, days_back: int | None = None) -> list[CandidateWork]:
        """Fetch EarthArXiv preprints via OAI-PMH ListRecords.

        Args:
            days_back: Number of days to look back (default from config).

        Returns:
            List of candidate works from EarthArXiv.
        """
        if days_back is None:
            days_back = self.config.days_back

        max_results = self.config.max_results

        # Calculate date range
        yesterday = utc_yesterday_end()
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        from_date = yesterday_start - timedelta(days=days_back - 1)

        logger.info(
            "Fetching EarthArXiv preprints from %s (max %d)",
            from_date.strftime("%Y-%m-%d"),
            max_results,
        )

        results: list[CandidateWork] = []
        resumption_token: str | None = None

        while len(results) < max_results:
            try:
                records, resumption_token = self._fetch_records(
                    from_date=from_date,
                    resumption_token=resumption_token,
                )
            except SourceFetchError:
                raise
            except Exception as e:
                logger.error("Error fetching EarthArXiv records: %s", e)
                break

            if not records:
                break

            for record in records:
                if len(results) >= max_results:
                    break

                work = self._parse_record(record)
                if work:
                    results.append(work)

            # No more pages
            if not resumption_token:
                break

        logger.info("Fetched %d EarthArXiv preprints", len(results))
        return results

    def _fetch_records(
        self,
        from_date: datetime,
        resumption_token: str | None = None,
    ) -> tuple[list[ET.Element], str | None]:
        """Fetch records from OAI-PMH endpoint.

        Args:
            from_date: Filter records from this date.
            resumption_token: Token for pagination (from previous response).

        Returns:
            Tuple of (list of record elements, next resumption token or None).
        """
        if resumption_token:
            # Use resumption token for subsequent requests
            params = {
                "verb": "ListRecords",
                "resumptionToken": resumption_token,
            }
        else:
            # Initial request with date filter
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc",
                "from": from_date.strftime("%Y-%m-%d"),
            }

        try:
            resp = self.session.get(
                EARTHARXIV_OAI_URL,
                params=params,
                timeout=DEFAULT_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise SourceFetchError(
                "eartharxiv",
                f"Request timed out after {DEFAULT_HTTP_TIMEOUT}s",
            ) from None
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            raise SourceFetchError("eartharxiv", f"HTTP {status} error") from e
        except requests.exceptions.RequestException as e:
            raise SourceFetchError(
                "eartharxiv",
                f"Network error: {type(e).__name__}",
            ) from e

        # Parse XML response
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            logger.error("Failed to parse OAI-PMH response: %s", e)
            return [], None

        # Check for OAI-PMH errors
        error = root.find(".//oai:error", OAI_NS)
        if error is not None:
            error_code = error.get("code", "unknown")
            error_msg = error.text or "Unknown error"
            if error_code == "noRecordsMatch":
                # No records in date range - not an error
                logger.info("No EarthArXiv records found in date range")
                return [], None
            logger.error("OAI-PMH error: %s - %s", error_code, error_msg)
            return [], None

        # Extract records
        records = root.findall(".//oai:record", OAI_NS)

        # Get resumption token for pagination
        next_token = None
        token_elem = root.find(".//oai:resumptionToken", OAI_NS)
        if token_elem is not None and token_elem.text:
            next_token = token_elem.text.strip()

        return records, next_token

    def _parse_record(self, record: ET.Element) -> CandidateWork | None:
        """Parse OAI-PMH record to CandidateWork.

        Args:
            record: XML Element containing the record.

        Returns:
            CandidateWork or None if parsing fails.
        """
        # Get Dublin Core metadata
        dc = record.find(".//oai_dc:dc", OAI_NS)
        if dc is None:
            return None

        # Title
        title_elem = dc.find("dc:title", OAI_NS)
        title = clean_title(title_elem.text if title_elem is not None else None)
        if not title:
            return None

        # Abstract/Description
        desc_elem = dc.find("dc:description", OAI_NS)
        abstract = (desc_elem.text or "").strip() if desc_elem is not None else None

        # Authors (multiple dc:creator elements)
        authors = []
        for creator in dc.findall("dc:creator", OAI_NS):
            if creator.text:
                authors.append(creator.text.strip())

        # DOI (from dc:identifier elements)
        doi = None
        url = None
        identifier = None
        for id_elem in dc.findall("dc:identifier", OAI_NS):
            if id_elem.text:
                text = id_elem.text.strip()
                if "doi.org/" in text:
                    doi = text.split("doi.org/")[-1]
                elif text.startswith("10."):
                    doi = text
                elif text.startswith("http"):
                    url = text
                elif text.isdigit():
                    identifier = text

        # Publication date (first dc:date)
        published = None
        date_elem = dc.find("dc:date", OAI_NS)
        if date_elem is not None and date_elem.text:
            try:
                date_str = date_elem.text.strip()
                if "T" in date_str:
                    published = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                else:
                    published = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass

        # Subjects/Keywords
        subjects = []
        for subj in dc.findall("dc:subject", OAI_NS):
            if subj.text:
                subjects.append(subj.text.strip())

        # Build URL if not found
        if not url and identifier:
            url = f"https://eartharxiv.org/repository/view/{identifier}/"

        # Use OAI identifier as fallback
        header = record.find(".//oai:header", OAI_NS)
        oai_id = None
        if header is not None:
            id_elem = header.find("oai:identifier", OAI_NS)
            if id_elem is not None and id_elem.text:
                oai_id = id_elem.text.strip()

        return CandidateWork(
            source="eartharxiv",
            identifier=oai_id or identifier or title,
            title=title,
            abstract=abstract,
            authors=authors,
            doi=doi,
            url=url,
            published=published,
            venue="EarthArXiv",
            extra={
                "subjects": subjects,
            },
        )


__all__ = ["EartharxivSource"]
