"""SQL Agent node — sends the user question to Databricks Genie for Text-to-SQL.

Genie generates SQL, executes it against the Delta table in Unity Catalog,
and returns structured results + natural language answer.

When Genie returns only an aggregate (COUNT / SUM), the agent rewrites
the SQL to SELECT actual facility rows and runs it directly via the
Databricks SQL Warehouse — guaranteeing real facility names in evidence.

Ref: https://docs.databricks.com/en/genie/
"""

import logging
import os
import re

import mlflow

from src.config import db_client
from src.data_loader import CITY_TO_REGION
from src.state import AgentState
from src.tools.genie_tool import query_genie

log = logging.getLogger(__name__)

# Columns to fetch when rewriting aggregate → detail query
_DETAIL_COLS = "name, region_normalized, facilityTypeId, address_city"


def _fill_region(row: list[str]) -> list[str]:
    """Fill missing region (index 1) using city (index 3) or facility name."""
    region = (row[1] or "").strip() if len(row) > 1 else ""
    if region and region.lower() not in ("", "unknown", "none", "null"):
        return row

    # Try city → region lookup
    city = (row[3] or "").strip() if len(row) > 3 else ""
    if city:
        for city_key, reg in CITY_TO_REGION.items():
            if city_key.lower() == city.lower():
                row[1] = reg
                return row

    # Try name-based inference
    name = (row[0] or "").strip() if row else ""
    name_lower = name.lower()
    for city_key, reg in CITY_TO_REGION.items():
        if len(city_key) >= 4 and city_key.lower() in name_lower:
            row[1] = reg
            return row

    # Default fallback
    row[1] = "Greater Accra"
    return row


def _is_aggregate_only(result: dict) -> bool:
    """Return True if Genie returned only a count/aggregate, not facility rows."""
    data = result.get("data", [])
    cols = [c.lower() for c in result.get("columns", [])]
    if len(data) <= 1 and len(cols) <= 2:
        count_keywords = {"count", "cnt", "total", "sum", "avg", "min", "max"}
        if any(kw in c for c in cols for kw in count_keywords):
            return True
        if len(data) == 1 and len(data[0]) == 1:
            try:
                int(data[0][0])
                return True
            except (ValueError, TypeError):
                pass
    return False


def _rewrite_count_to_select(sql: str) -> str | None:
    """Rewrite a SELECT COUNT(*) ... query into SELECT name, region, type, city ...

    Returns the rewritten SQL or None if the SQL can't be parsed.
    """
    if not sql:
        return None
    # Match: SELECT COUNT(*) ... FROM ...
    # Replace the SELECT ... FROM with SELECT detail cols FROM, keep WHERE/etc.
    pattern = re.compile(
        r"SELECT\s+COUNT\s*\([^)]*\).*?FROM",
        re.IGNORECASE | re.DOTALL,
    )
    if not pattern.search(sql):
        return None
    rewritten = pattern.sub(f"SELECT {_DETAIL_COLS} FROM", sql, count=1)
    # Remove any GROUP BY / ORDER BY that referenced the count
    rewritten = re.sub(r"\bORDER\s+BY\s+.*$", "", rewritten, flags=re.IGNORECASE)
    # Add a LIMIT to avoid huge results
    if "LIMIT" not in rewritten.upper():
        rewritten = rewritten.rstrip().rstrip(";") + " LIMIT 30"
    return rewritten


def _run_sql_direct(sql: str) -> tuple[list[list[str]], list[str]]:
    """Execute SQL directly on Databricks SQL Warehouse and return (rows, columns)."""
    catalog = os.getenv("DATABRICKS_CATALOG", "virtue_foundation")
    schema = os.getenv("DATABRICKS_SCHEMA", "ghana_medical")

    warehouses = list(db_client.warehouses.list())
    if not warehouses:
        return [], []
    wh_id = warehouses[0].id

    from databricks.sdk.service.sql import Disposition

    resp = db_client.statement_execution.execute_statement(
    warehouse_id=wh_id,
    statement=sql,
    catalog=catalog,
    schema=schema,
    wait_timeout="30s",
    disposition=Disposition.INLINE,
    )
    rows = []
    columns = []
    if resp.result and resp.result.data_array:
        rows = resp.result.data_array
    if resp.manifest and resp.manifest.schema:
        columns = [c.name for c in resp.manifest.schema.columns]
    return rows, columns

def _build_fallback_sql(query: str) -> str:
    """Build direct SQL from common query patterns when Genie times out."""
    catalog = os.getenv("DATABRICKS_CATALOG", "virtue_foundation")
    schema = os.getenv("DATABRICKS_SCHEMA", "ghana_medical")
    table = f"{catalog}.{schema}.ghana_facilities"
    q = query.lower()

    specialties = [
        "cardiology", "ophthalmology", "pediatrics", "generalSurgery",
        "emergencyMedicine", "gynecologyAndObstetrics", "orthopedicSurgery",
        "internalMedicine", "familyMedicine", "dentistry", "radiology",
        "surgery", "neurology", "oncology", "dermatology", "urology"
    ]

    for spec in specialties:
        if spec.lower() in q:
            if any(k in q for k in ["how many", "count", "number"]):
                return f"SELECT COUNT(*) as count FROM {table} WHERE specialties LIKE '%{spec}%'"
            elif any(k in q for k in ["region", "where", "which area"]):
                return f"""
                    SELECT region_normalized, COUNT(*) as count
                    FROM {table}
                    WHERE specialties LIKE '%{spec}%'
                    GROUP BY region_normalized
                    ORDER BY count DESC
                """
            else:
                return f"""
                    SELECT name, region_normalized, facilityTypeId, address_city
                    FROM {table}
                    WHERE specialties LIKE '%{spec}%'
                    LIMIT 20
                """

    if any(k in q for k in ["how many", "count"]):
        if "hospital" in q:
            return f"SELECT COUNT(*) as count FROM {table} WHERE facilityTypeId = 'hospital'"
        if "clinic" in q:
            return f"SELECT COUNT(*) as count FROM {table} WHERE facilityTypeId = 'clinic'"
        return f"SELECT facilityTypeId, COUNT(*) as count FROM {table} GROUP BY facilityTypeId ORDER BY count DESC"

    if "region" in q and any(k in q for k in ["most", "highest", "top"]):
        return f"""
            SELECT region_normalized, COUNT(*) as facility_count
            FROM {table}
            WHERE region_normalized IS NOT NULL
            GROUP BY region_normalized
            ORDER BY facility_count DESC
            LIMIT 10
        """

    return f"""
        SELECT name, region_normalized, facilityTypeId, address_city
        FROM {table}
        WHERE facilityTypeId IS NOT NULL
        LIMIT 20
    """

@mlflow.trace(name="sql_agent_node", span_type="AGENT")
def sql_agent_node(state: AgentState) -> dict:
    """SQL Agent — forwards query to Genie, falls back to direct SQL on timeout."""
    query = state["query"]
    result = None
    used_fallback = False

    # Step 1: Try Genie
    try:
        result = query_genie(query)
    except Exception as e:
        log.warning("Genie failed: %s — switching to direct SQL", e)

    # Step 2: If Genie timed out or returned nothing, use direct SQL
    if not result or (not result.get("data") and not result.get("text")):
        log.info("Genie returned no data — running direct SQL fallback")
        used_fallback = True
        fallback_sql = _build_fallback_sql(query)
        try:
            rows, cols = _run_sql_direct(fallback_sql)
            result = {
                "sql": fallback_sql,
                "columns": cols,
                "data": rows,
                "text": f"Direct SQL returned {len(rows)} result(s).",
                "description": "Direct SQL fallback (Genie timed out)",
            }
        except Exception as e:
            log.warning("Direct SQL fallback also failed: %s", e)
            result = result or {}

    # Step 3: If we got aggregate only, rewrite to get facility rows
    if not used_fallback and _is_aggregate_only(result):
        log.info("Aggregate-only result — rewriting SQL to fetch facility details")
        original_sql = result.get("sql", "")
        detail_sql = _rewrite_count_to_select(original_sql)

        if detail_sql:
            try:
                rows, cols = _run_sql_direct(detail_sql)
                if rows:
                    seen = set()
                    unique_rows = []
                    for row in rows:
                        key = row[0] if row else ""
                        if key and key not in seen:
                            seen.add(key)
                            unique_rows.append(row)
                    unique_rows = [_fill_region(list(r)) for r in unique_rows]
                    result["detail_data"] = unique_rows
                    result["detail_columns"] = cols
                    log.info("Got %d unique facility rows from rewritten SQL", len(unique_rows))
            except Exception as e:
                log.warning("Direct SQL execution failed: %s", e)

        if not result.get("detail_data"):
            log.info("SQL rewrite didn't produce results — trying Genie follow-up")
            try:
                detail_query = f"List the names, regions, and types of the facilities for: {query}"
                detail_result = query_genie(detail_query)
                if detail_result.get("data"):
                    result["detail_data"] = detail_result["data"]
                    result["detail_columns"] = detail_result.get("columns", [])
            except Exception as e:
                log.warning("Genie follow-up failed: %s", e)

    return {
        "sql_result": result,
        "citations": [{"source": "genie" if not used_fallback else "direct_sql",
                       "sql": result.get("sql"),
                       "description": result.get("description")}],
    }
