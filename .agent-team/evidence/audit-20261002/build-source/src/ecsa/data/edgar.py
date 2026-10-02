"""SEC EDGAR metadata, earnings exhibits and conservative MD&A extraction."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ecsa.data.cache import DiskCache, HttpClient
from ecsa.data.transcripts import parse_transcript, session_timing
from ecsa.schemas import DocType, Document, Section, Utterance
from ecsa.settings import Settings
from ecsa.storage import utc


def pad_cik(cik: str | int) -> str:
    """Validate and normalize CIK to ten decimal digits."""
    value = str(cik).strip()
    if not re.fullmatch(r"[0-9]{1,10}", value):
        raise ValueError("CIK must contain 1-10 digits")
    return value.zfill(10)


@dataclass(frozen=True)
class Filing:
    """Filing identity; acceptance_datetime is the UTC availability, if supplied by SEC."""

    cik: str
    accession: str
    form: str
    filing_date: date
    acceptance_datetime: datetime | None
    items: tuple[str, ...]
    primary_document: str

    @property
    def base_url(self) -> str:
        """SEC archive directory for this filing."""
        accession = self.accession.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{accession}/"


def parse_submissions(payload: dict[str, Any], cik: str) -> list[Filing]:
    """Parse recent and paginated column-oriented SEC submissions JSON."""
    rows = payload.get("filings", {}).get("recent", payload)
    output = []
    for i, accession in enumerate(rows.get("accessionNumber", [])):

        def value(key: str, default: str = "") -> str:
            values = rows.get(key, [])
            return str(values[i]) if i < len(values) and values[i] is not None else default

        timestamp = value("acceptanceDateTime")
        output.append(
            Filing(
                pad_cik(cik),
                str(accession),
                value("form"),
                date.fromisoformat(value("filingDate")),
                utc(timestamp) if timestamp else None,
                tuple(re.findall(r"\d+\.\d+", value("items"))),
                value("primaryDocument"),
            )
        )
    return output


def extract_mdna(html: str, form: str) -> str | None:
    """Extract 10-Q Item 2 / 10-K Item 7 only with unambiguous heading boundaries."""
    if form not in {"10-Q", "10-K"}:
        raise ValueError("MD&A supports 10-Q or 10-K")
    soup = BeautifulSoup(html, "lxml")
    for element in soup(["script", "style", "ix:header"]):
        element.decompose()
    text = soup.get_text("\n", strip=True).replace("\xa0", " ")
    item = "2" if form == "10-Q" else "7"
    stop = "3|4" if form == "10-Q" else "7A|8"
    starts = list(
        re.finditer(
            rf"(?im)^\s*item\s+{item}\s*[.:-]?\s*management[’']?s?\s+discussion(?:\s+and\s+analysis)?",
            text,
        )
    )
    candidates = []
    for start in starts:
        end = re.search(rf"(?im)^\s*item\s+(?:{stop})\b", text[start.end() :])
        if not end:
            continue
        candidate = text[start.start() : start.end() + end.start()].strip()
        # Exclude terse table-of-contents entries; multiple plausible sections are ambiguous.
        if len(candidate.split()) >= 50 and not re.search(
            r"(?im)^\s*item\s+\d+", candidate[start.end() - start.start() :]
        ):
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


class EdgarClient:
    """SEC client with declared identity, persistent cache and <10 requests/second."""

    def __init__(self, settings: Settings, http: HttpClient | None = None) -> None:
        identity = settings.sec_user_agent
        agent = identity.get_secret_value() if identity else ""
        if not re.fullmatch(r".+\s+[^\s@]+@[^\s@]+\.[^\s@]+", agent):
            raise ValueError("SEC_USER_AGENT must be 'Name email' in the environment")
        self.http = http or HttpClient(
            DiskCache(settings.cache_dir / "sec", settings.cache_ttl_seconds),
            rate=settings.sec_requests_per_second,
            retries=settings.request_max_retries,
            timeout=settings.request_timeout_seconds,
            backoff=settings.retry_backoff_seconds,
            headers={"User-Agent": agent},
        )

    def ticker_to_cik(self, ticker: str) -> str:
        """Map ticker case-insensitively to a zero-padded ten-digit CIK."""
        rows = self.http.json("https://www.sec.gov/files/company_tickers.json")
        for record in rows.values():
            if record["ticker"].upper() == ticker.upper():
                return pad_cik(record["cik_str"])
        raise KeyError(f"No SEC CIK for {ticker}")

    def list_filings(self, cik: str, forms: list[str], start: date, end: date) -> list[Filing]:
        """List filings by filing date, including overlapping older submissions pages."""
        cik = pad_cik(cik)
        payload = self.http.json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        filings = parse_submissions(payload, cik)
        for page in payload.get("filings", {}).get("files", []):
            if (
                page.get("filingFrom", "0001-01-01") <= end.isoformat()
                and page.get("filingTo", "9999-12-31") >= start.isoformat()
            ):
                filings += parse_submissions(
                    self.http.json("https://data.sec.gov/submissions/" + page["name"]), cik
                )
        unique = {
            f.accession: f for f in filings if f.form in forms and start <= f.filing_date <= end
        }
        return sorted(unique.values(), key=lambda f: (f.filing_date, f.accession))

    def earnings_8ks(
        self, cik: str, start: date, end: date, ticker: str = "UNKNOWN"
    ) -> list[Document]:
        """Download EX-99.x exhibits only for 8-K Item 2.02, using SEC availability."""
        result = []
        for filing in self.list_filings(cik, ["8-K", "8-K/A"], start, end):
            if "2.02" not in filing.items or filing.acceptance_datetime is None:
                continue
            index = self.http.request(filing.base_url + filing.accession + "-index.html")
            for row in BeautifulSoup(index, "lxml").select("table.tableFile tr"):
                cells = row.find_all("td")
                if len(cells) < 4 or not re.fullmatch(
                    r"EX-99(?:\.\d+)?", cells[3].get_text(strip=True), re.I
                ):
                    continue
                anchor = cells[2].find("a")
                if anchor is None:
                    continue
                url = urljoin(filing.base_url, str(anchor.get("href", "")))
                data = self.http.request(url)
                description = cells[1].get_text(" ", strip=True).lower()
                kind: DocType = (
                    "presentation"
                    if re.search("slide|presentation", description)
                    else "transcript"
                    if "transcript" in description
                    else "8k_press_release"
                )
                if data.startswith(b"%PDF"):
                    from pypdf import PdfReader

                    text = "\n".join(
                        page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages
                    )
                else:
                    text = BeautifulSoup(data, "lxml").get_text("\n", strip=True)
                if text.strip():
                    result.append(
                        Document(
                            ticker=ticker,
                            cik=filing.cik,
                            doc_type=kind,
                            event_id=f"{ticker}:{filing.acceptance_datetime.isoformat()}",
                            event_datetime=filing.acceptance_datetime,
                            available_at=filing.acceptance_datetime,
                            session_timing=session_timing(filing.acceptance_datetime),
                            source="SEC_EDGAR",
                            url=url,
                            raw_text=text,
                            sections=parse_transcript(text),
                        )
                    )
        return result

    def mdna(
        self,
        cik: str,
        form: str,
        *,
        start: date = date(1900, 1, 1),
        end: date | None = None,
        ticker: str = "UNKNOWN",
    ) -> Document | None:
        """Extract the latest matching filing's MD&A; return None when ambiguous/missing."""
        filings = self.list_filings(cik, [form], start, end or date.today())
        if not filings:
            return None
        filing = filings[-1]
        if filing.acceptance_datetime is None:
            return None
        url = filing.base_url + filing.primary_document
        text = extract_mdna(self.http.request(url).decode("utf-8", errors="replace"), form)
        if text is None:
            return None
        return Document(
            ticker=ticker,
            cik=filing.cik,
            doc_type="10q_mdna" if form == "10-Q" else "10k_mdna",
            event_datetime=filing.acceptance_datetime,
            available_at=filing.acceptance_datetime,
            session_timing=session_timing(filing.acceptance_datetime),
            source="SEC_EDGAR",
            url=url,
            raw_text=text,
            sections=[
                Section(
                    name="mdna",
                    utterances=[Utterance(speaker="Management", role="executive", text=text)],
                )
            ],
        )
