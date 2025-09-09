"""
Txt2SQL + iCal ingestion utilities for the c-base bot.

- Imports the c-base iCal feed into DuckDB (run once at server start).
- Provides a Txt2SQLEnricher FrameProcessor that:
    • detects calendar-ish questions,
    • generates DuckDB SQL with a local Ollama model,
    • executes safely against the calendar DB,
    • injects a compact "DB Context" system message (with SPEAK_THIS) for the LLM to use.

Requires:
  pip install duckdb requests icalendar pytz python-dateutil pandas loguru
"""

from __future__ import annotations

import os
import re
import json
import http.client
import urllib.parse
import random
from datetime import datetime, date, timedelta
from typing import Optional, Set, Tuple, List

import duckdb
import requests
import icalendar
import pytz
import pandas as pd
from dateutil.rrule import rrulestr
from dateutil.parser import parse
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TranscriptionFrame, TextFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame

from typing import List, Set, Optional
import duckdb
import requests
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TranscriptionFrame, TextFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# Defaults (can be overridden via env or in bot.py)
CALENDAR_DB_PATH = os.environ.get("CALENDAR_DB_PATH", "calendar.duckdb")
ICAL_URL = os.environ.get("ICAL_URL", "https://www.c-base.org/calendar/exported/c-base-events.ics")

# Grounding policy injected by bot.py so the LLM obeys DB results
DB_POLICY = (
    "Grounding policy: When a message labeled 'DB Context' is present, you must answer "
    "STRICTLY and ONLY using the facts in that DB Context. If it contains 'SPEAK_THIS:', "
    "repeat that text verbatim as your entire answer. If DB Context is missing or empty, "
    "say you don't have calendar data. Never invent events, dates, or people."
)

# -----------------------
# iCal → DuckDB ingestion
# -----------------------

def download_ical(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"Could not download iCal: {e}")
        return None


def parse_ical_to_duckdb(ical_content: Optional[str], db_path: str = CALENDAR_DB_PATH) -> bool:
    if not ical_content:
        logger.warning("No iCal content; skipping calendar import.")
        return False

    conn = duckdb.connect(db_path)

    # Drop and recreate table to avoid stale data
    conn.execute("DROP TABLE IF EXISTS events")
    conn.execute(
        """
        CREATE TABLE events (
            uid STRING,
            summary STRING,
            dtstart TIMESTAMP,
            dtend TIMESTAMP,
            dtstamp TIMESTAMP,
            location STRING,
            description STRING,
            status STRING,
            sequence INTEGER,
            created TIMESTAMP,
            last_modified TIMESTAMP,
            pate STRING,
            UNIQUE(summary, dtstart, pate, location)
        )
        """
    )

    try:
        cal = icalendar.Calendar.from_ical(ical_content)
    except Exception as e:
        logger.error(f"Error parsing iCal content: {e}")
        conn.close()
        return False

    event_count = 0
    recurrence_count = 0
    skipped_count = 0

    for event in cal.walk('VEVENT'):
        uid = str(event.get('UID', f"event_{event_count}"))
        summary = str(event.get('SUMMARY', ''))

        dtstart_raw = event.get('DTSTART')
        rrule_raw = event.get('RRULE')
        dtstart = dtstart_raw.dt if dtstart_raw else None

        dtend = event.get('DTEND').dt if event.get('DTEND') else dtstart
        dtstamp = event.get('DTSTAMP').dt if event.get('DTSTAMP') else None
        created = event.get('CREATED').dt if event.get('CREATED') else None
        last_modified = event.get('LAST-MODIFIED').dt if event.get('LAST-MODIFIED') else None

        # Normalize to UTC
        def to_utc(x):
            if isinstance(x, datetime) and x.tzinfo:
                return x.astimezone(pytz.UTC)
            return x

        dtstart = to_utc(dtstart)
        dtend = to_utc(dtend)
        dtstamp = to_utc(dtstamp)
        created = to_utc(created)
        last_modified = to_utc(last_modified)

        # Handle date-only events (treat as all-day)
        if isinstance(dtstart, date) and not isinstance(dtstart, datetime):
            dtstart = datetime.combine(dtstart, datetime.min.time(), tzinfo=pytz.UTC)
        if isinstance(dtend, date) and not isinstance(dtend, datetime):
            dtend = datetime.combine(dtend, datetime.min.time(), tzinfo=pytz.UTC) + timedelta(days=1) - timedelta(seconds=1)

        # Skip non-recurring events before 2024, but process recurring ones
        if not rrule_raw and dtstart and dtstart.year < 2024:
            skipped_count += 1
            continue

        location = str(event.get('LOCATION', ''))
        description = str(event.get('DESCRIPTION', ''))
        status = str(event.get('STATUS', ''))
        sequence = int(event.get('SEQUENCE', 0))

        # Extract pate: name from description
        pate_match = re.search(r'pate:\s*([\w\s,]+)', description, re.IGNORECASE)
        pate = pate_match.group(1).strip() if pate_match else ''

        # Insert base event only if it's from 2024 or later (skip old bases for recurring events)
        if dtstart and dtstart.year >= 2024:
            try:
                conn.execute(
                    """
                    INSERT INTO events (
                        uid, summary, dtstart, dtend, dtstamp, location,
                        description, status, sequence, created, last_modified, pate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uid, summary, dtstart, dtend, dtstamp, location,
                        description, status, sequence, created, last_modified, pate,
                    ),
                )
                event_count += 1  # Increment only if base is inserted
            except duckdb.ConstraintException:
                pass

        # Handle recurring events (RRULE) - always process if present, even for old originals
        if rrule_raw and dtstart:
            try:
                rrule_str = rrule_raw.to_ical().decode('utf-8')

                # Check for UNTIL and remove if in the past (assuming ongoing events with outdated iCal)
                now = datetime.now(pytz.UTC)
                rrule_parts = rrule_str.split(';')
                new_parts = []
                has_until = False
                until = None
                for p in rrule_parts:
                    if p.startswith('UNTIL='):
                        has_until = True
                        until_str = p[6:]
                        try:
                            until = parse(until_str)
                            if until.tzinfo is None:
                                until = until.replace(tzinfo=pytz.UTC)
                        except Exception as parse_e:
                            logger.warning(f"Failed to parse UNTIL '{until_str}' for {summary}: {parse_e}")
                            until = None
                    else:
                        new_parts.append(p)
                if has_until and until and until < now:
                    logger.info(f"Ignoring past UNTIL for ongoing event '{summary}'")
                    rrule_str = ';'.join(new_parts)

                rrule = rrulestr(rrule_str, dtstart=dtstart)

                # Generate recurring events from now through next 2 years
                end_date = now + timedelta(days=730)  # 2 years from now

                # For recurring events, always generate from 3 months ago regardless of original start date
                # This ensures old weekly/monthly events still show up as current recurring instances
                search_start = now - timedelta(days=90)

                for recurrence in rrule.between(search_start, end_date, inc=True):
                    # Only include events from 2024 onwards to avoid very old events
                    if recurrence.year < 2024:
                        skipped_count += 1
                        continue
                    recurrence_dtstart = recurrence
                    duration = (dtend - dtstart) if (dtend and dtstart) else timedelta(hours=1)
                    recurrence_dtend = recurrence_dtstart + duration
                    recurrence_uid = f"{uid}_{recurrence_dtstart.strftime('%Y%m%dT%H%M%SZ')}"
                    try:
                        conn.execute(
                            """
                            INSERT INTO events (
                                uid, summary, dtstart, dtend, dtstamp, location,
                                description, status, sequence, created, last_modified, pate
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                recurrence_uid, summary, recurrence_dtstart, recurrence_dtend, dtstamp, location,
                                description, status, sequence, created, last_modified, pate,
                            ),
                        )
                        recurrence_count += 1
                    except duckdb.ConstraintException:
                        pass
            except Exception as e:
                logger.warning(f"RRULE parse failed for {summary}: {e}")

    conn.commit()
    conn.close()
    logger.info(
        f"Calendar import complete: {event_count} base events, {recurrence_count} recurrences, skipped {skipped_count} old events."
    )
    return True


def init_calendar_db_from_ical(url: str = ICAL_URL, db_path: str = CALENDAR_DB_PATH) -> None:
    ical_text = download_ical(url)
    ok = parse_ical_to_duckdb(ical_text, db_path)
    if not ok:
        # Ensure the DB exists with an empty events table to keep downstream components happy
        try:
            con = duckdb.connect(db_path)
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    uid STRING,
                    summary STRING,
                    dtstart TIMESTAMP,
                    dtend TIMESTAMP,
                    dtstamp TIMESTAMP,
                    location STRING,
                    description STRING,
                    status STRING,
                    sequence INTEGER,
                    created TIMESTAMP,
                    last_modified TIMESTAMP,
                    pate STRING
                )
                """
            )
            con.close()
            logger.info("Created empty events table as fallback.")
        except Exception as e:
            logger.error(f"Failed to create fallback events table: {e}")

# -----------------------
# NL → SQL for DuckDB
# -----------------------

SQL_ONLY_RE = re.compile(r"^\s*(WITH\b|SELECT\b)", re.IGNORECASE | re.DOTALL)
FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|ATTACH|DETACH|CREATE|REPLACE|PRAGMA|COPY|LOAD|EXPORT|SET)\b",
    re.IGNORECASE,
)
FROMJOIN_FIND = re.compile(r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.IGNORECASE)


class OllamaChat:
    """Tiny chat client for local Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gpt-oss:20b", timeout: int = 30):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        url = urllib.parse.urlparse(self.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=self.timeout)
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            }
        )
        conn.request("POST", "/api/chat", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "")


def duckdb_schema_summary(db_path: str, allow_tables: Optional[Set[str]] = None) -> str:
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT table_schema, table_name, list(col_name || ' ' || col_type) AS cols
            FROM (
              SELECT table_schema, table_name, column_name AS col_name, data_type AS col_type
              FROM information_schema.columns
              WHERE table_schema NOT IN ('pg_catalog','information_schema')
            )
            GROUP BY 1,2
            ORDER BY 1,2
            """
        ).fetchall()
    finally:
        con.close()
    lines = []
    for sch, tbl, cols in rows:
        fq = f"{sch}.{tbl}" if sch and sch != 'main' else tbl
        if allow_tables and fq not in allow_tables and tbl not in allow_tables:
            continue
        coltxt = ", ".join(cols)
        lines.append(f"- {fq}({coltxt})")
    return "\n".join(lines) if lines else "(no visible tables)"


def harden_sql(sql: str, allow_tables: Optional[Set[str]], default_limit: int = 200) -> str:
    s = sql.strip().strip("`").strip()
    s = s.strip("```").replace("sql\n", "").replace("SQL:", "").strip()
    if not SQL_ONLY_RE.search(s):
        raise ValueError("Only SELECT/CTE queries are allowed.")
    if FORBIDDEN_RE.search(s):
        raise ValueError("DDL/DML detected. Rejected.")
    if allow_tables:
        tables = set(m.group(1).split()[-1] for m in FROMJOIN_FIND.finditer(s))
        norm = set()
        for t in tables:
            norm.add(t)
            if "." in t:
                norm.add(t.split(".")[-1])
        if not norm.issubset(set(allow_tables)):
            bad = ", ".join(sorted(norm - set(allow_tables)))
            raise ValueError(f"Unknown/disallowed table(s): {bad}")
    if re.search(r"\blimit\s+\d+\b", s, re.IGNORECASE) is None:
        s = s.rstrip(";") + f"\nLIMIT {default_limit};"
    return s

SYS_NL2SQL = """You translate a natural-language question into ONE DuckDB-compatible SQL SELECT statement.
Rules:
- SELECT or CTEs + SELECT only. No writes, no PRAGMA, no configuration.
- Use only columns/tables from the provided schema summary.
- Prefer explicit joins. Prefer ISO timestamps if needed.
- Add an ORDER BY when it improves readability.
- If the question is ambiguous, choose the simplest reasonable interpretation.
- Return ONLY the SQL, no prose, no fencing.
"""

def user_nl2sql_prompt(question: str, schema_summary: str) -> str:
    fewshots = """
Q: Count events per day for the next 7 days.
SQL:
WITH days AS (
  SELECT (current_date + i) AS d FROM range(7) t(i)
)
SELECT d AS day, COUNT(*) AS events
FROM days
LEFT JOIN events e ON date(e.dtstart) = d
GROUP BY 1
ORDER BY 1;

Q: List the next 10 upcoming events with title and start time.
SQL:
SELECT summary, dtstart
FROM events
WHERE dtstart >= now()
ORDER BY dtstart ASC
LIMIT 10;
"""
    return f"""Database schema:
{schema_summary}

{fewshots}

Task: Write ONE SQL query for:
"{question}"
"""

def run_duckdb(db_path: str, sql: str) -> Tuple[List[str], List[tuple]]:
    con = duckdb.connect(db_path, read_only=True, config={"threads": "1"})
    try:
        con.execute(f"EXPLAIN {sql}")
        res = con.execute(sql)
        cols = [d[0] for d in res.description]
        rows = res.fetchall()
        return cols, rows
    finally:
        con.close()

# Canned responses for immediate feedback when processing calendar queries
CALENDAR_CANNED_RESPONSES = [
    "One moment while I sync with the station chronolog.",
    "Hold on—querying the temporal index.",
    "Give me a second to ping the orbital scheduler.",
    "Stand by; consulting the nav-com event stack.",
    "Brief pause while I align with the local time grid.",
    "Hold tight—cross-checking the mission timetable.",
    "One tick while I pull the calendar from cold storage.",
    "Moment—interrogating the ship's log for upcoming slots.",
    "Sec—resolving conflicts in the appointment matrix.",
    "Wait a beat while I scan the sector schedule.",
    "Stand by—syncing with the quantum calendar.",
    "Let me tap the chronometer and see what's actually scheduled."
]

DB_KEYWORDS = [
    "query", "table", "count", "list", "show", "events", "calendar",
    "where", "group by", "order by", "join", "avg", "sum", "top", "latest", "upcoming",
]

def looks_like_db_question(text: str) -> bool:
    t = text.lower()
    # strong calendar hints
    if any(w in t for w in ["calendar", "event", "events", "schedule", "what's on", "whats on", "happening"]):
        return True
    # temporal cues that usually imply calendar lookup
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "tonight", "today", "tomorrow", "weekend", "next", "upcoming"]
    if any(d in t for d in DAYS):
        return True
    hits = sum(1 for k in DB_KEYWORDS if k in t)
    return hits >= 2

# -----------------------
# Result formatter for TTS
# -----------------------

def _format_rows_for_speech(cols: List[str], rows: List[tuple], max_items: int = 3) -> str:
    """Build a short, TTS-friendly string from (cols, rows)."""
    cest = pytz.timezone("Europe/Berlin")
    if not rows:
        return "No matching events found."

    df = pd.DataFrame(rows, columns=cols)
    parts = []

    def safe(val, default=""):
        if pd.isna(val) or val is None:
            return default
        s = str(val).strip()
        return s if s else default

    for _, r in df.head(max_items).iterrows():
        summary = safe(r.get("summary"))
        loc = safe(r.get("location"), "no specific location")
        pate = safe(r.get("pate"), "nobody specific")
        desc = safe(r.get("description"))

        # time formatting
        when = "On an unspecified date"
        dtstart_val = r.get("dtstart")
        try:
            ts = pd.to_datetime(dtstart_val, utc=True)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(cest)
            date_str = ts.strftime("%B %d, %Y")
            time_str = ts.strftime("%I:%M %p").lstrip("0")
            when = f"On {date_str} at {time_str}"
        except Exception:
            pass

        # clean description and clamp
        desc = re.sub(r"pate:\s*[\w\s,]+", "", desc, flags=re.IGNORECASE).strip()
        if not desc:
            desc = "no extra details available"
        if len(desc) > 160:
            desc = desc[:160] + "..."

        parts.append(f"{when}, there's {summary}, hosted by {pate} at {loc}. {desc.capitalize()}.")

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        second = parts[1][0].lower() + parts[1][1:]
        return f"{parts[0]} Then, {second}"
    return "There are a few things going on. " + " ".join(parts)

# -----------------------
# FrameProcessor
# -----------------------

"""Simple pass-through processor that intercepts TranscriptionFrames for calendar queries
and enriches the LLM context with database results.
"""

# Calendar enrichment function that can be called directly
async def enrich_calendar_query(
    query: str, 
    context: OpenAILLMContext,
    db_path: str = CALENDAR_DB_PATH,
    allow_tables: Set[str] = {"events"},
    ollama_url: str = "http://localhost:11434",
    model: str = "gpt-oss:20b",
    default_limit: int = 200,
    push_frame_func = None
) -> bool:
    """Process a calendar query and enrich the context with results.
    
    Returns True if this was a calendar query that was processed, False otherwise.
    """
    if not looks_like_db_question(query):
        return False
    
    logger.info(f"🗓️ txt2sql: processing calendar query '{query}'")
    
    # Send immediate canned response if push_frame function provided
    if push_frame_func:
        canned_response = random.choice(CALENDAR_CANNED_RESPONSES)
        logger.info(f"🎯 txt2sql: sending immediate response: '{canned_response}'")
        await push_frame_func(TextFrame(canned_response))
    
    # Generate and execute SQL query
    try:
        # Simple schema for txt2sql generation
        schema_summary = """
        Table: events
        Columns: 
        - summary (TEXT): event title
        - description (TEXT): event description
        - dtstart (TIMESTAMP): event start time
        - dtend (TIMESTAMP): event end time  
        - location (TEXT): event location
        - pate (TEXT): event organizer/host
        """
        
        # Generate SQL using Ollama
        prompt = user_nl2sql_prompt(query, schema_summary)
        
        response = requests.post(
            f"{ollama_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYS_NL2SQL},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 300
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            raw_sql = result["choices"][0]["message"]["content"].strip()
            
            # Clean and validate SQL
            sql_lines = [line.strip() for line in raw_sql.split('\n') if line.strip()]
            sql = '\n'.join(sql_lines)
            
            # Remove markdown fencing if present
            sql = re.sub(r'^```.*?sql\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
            sql = re.sub(r'```\s*$', '', sql)
            sql = sql.strip()
            
            logger.info(f"Generated SQL: {sql}")
            
            # Validate and execute
            try:
                # Simple validation - ensure it's a SELECT and uses allowed tables
                if not sql.upper().strip().startswith('SELECT'):
                    raise ValueError("Only SELECT queries allowed")
                
                # Check for allowed tables (simple check)
                tables_used = set(re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE))
                tables_used.update(re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE))
                
                if tables_used and not tables_used.issubset(allow_tables):
                    disallowed = tables_used - allow_tables
                    raise ValueError(f"Unknown/disallowed table(s): {', '.join(disallowed)}")
                    
                # Add LIMIT if not present
                if not re.search(r'\bLIMIT\s+\d+', sql, re.IGNORECASE):
                    sql = sql.rstrip(';') + f' LIMIT {default_limit};'
                
                cols, rows = run_duckdb(db_path, sql)
                
                if rows:
                    # Format results for speech
                    speech_text = _format_rows_for_speech(cols, rows, max_items=5)
                    context_msg = f"DB Context:\nSPEAK_THIS: {speech_text}"
                    logger.info(f"✅ txt2sql: found {len(rows)} rows, adding to context")
                else:
                    context_msg = "DB Context:\nSPEAK_THIS: No events found matching your query."
                    logger.info("📭 txt2sql: no results found")
                    
                context.add_message({"role": "system", "content": context_msg})
                return True
                
            except ValueError as ve:
                error_msg = f"DB Context:\nSPEAK_THIS: I don't have calendar data for that right now.\nERROR: {ve}"
                context.add_message({"role": "system", "content": error_msg})
                logger.warning(f"txt2sql failed: {ve}")
                return True
            except Exception as e:
                error_msg = f"DB Context:\nSPEAK_THIS: Calendar system temporarily unavailable.\nERROR: {e}"
                context.add_message({"role": "system", "content": error_msg})
                logger.error(f"txt2sql error: {e}")
                return True
        else:
            logger.error(f"Ollama request failed: {response.status_code}")
            return True
            
    except Exception as e:
        logger.error(f"txt2sql processing error: {e}")
        return True


__all__ = [
    "CALENDAR_DB_PATH",
    "ICAL_URL",
    "DB_POLICY",
    "init_calendar_db_from_ical",
    "enrich_calendar_query",
]
