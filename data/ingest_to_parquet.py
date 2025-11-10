#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime
import io
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from google.cloud import secretmanager, storage
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import (Boolean, Column, DateTime, Integer, Numeric, String,
                        Table, Text)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# ----------------- DB engine via Cloud SQL Connector -----------------

def create_engine_with_connector(
    instance_connection_name: str,
    db_name: str,
    db_user: str,
    enable_private_ip: bool = False,
    enable_iam_auth: bool = True,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> sa.Engine:
    connector = Connector()

    def getconn():
        ip_type = IPTypes.PRIVATE if enable_private_ip else IPTypes.PUBLIC
        return connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            db=db_name,
            enable_iam_auth=enable_iam_auth,
            ip_type=ip_type,
        )

    engine = sa.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        future=True,
    )
    return engine


# ----------------- Schema (Core Tables) -----------------

def reflect_or_define_tables(metadata: sa.MetaData) -> Dict[str, Table]:
    journal_entry = Table(
        "journal_entry",
        metadata,
        Column("journal_entry_id", PG_UUID, primary_key=True),
        Column("business_id", Text, nullable=False),
        Column("year_month", Integer, nullable=False),
        Column("date", DateTime(timezone=True), nullable=False),
        Column("number", Integer, nullable=False),
        Column("description", Text, nullable=False),
        Column("currency", String, nullable=False),
        Column("journal_entry_type", Text, nullable=False),
        Column("journal_entry_sub_type", Text, nullable=True),
        Column("journal_entry_status", sa.types.UserDefinedType(), nullable=False),
        Column("journal_entry_origin", sa.types.UserDefinedType(), nullable=False),
    )

    entry_line = Table(
        "entry_line",
        metadata,
        Column("entry_line_id", PG_UUID, primary_key=True),
        Column("journal_entry_id", PG_UUID, nullable=False),
        Column("ledger_account_id", PG_UUID, nullable=True),
        Column("index", Integer, nullable=False),
        Column("description", Text, nullable=False),
        Column("debit", Numeric, nullable=False),
        Column("credit", Numeric, nullable=False),
        Column("exchange_rate", Numeric, nullable=False),
    )

    ledger_account = Table(
        "ledger_account",
        metadata,
        Column("ledger_account_id", PG_UUID, primary_key=True),
        Column("business_id", Text, nullable=False),
        Column("number", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("parent_ledger_account_id", PG_UUID, nullable=True),
        Column("currency", String, nullable=False),
        Column("nature", String, nullable=False),
        Column("ledger_account_type", Text, nullable=True),
        Column("ledger_account_sub_type", Text, nullable=True),
        Column("ledger_account_sub_sub_type", Text, nullable=True),
        Column("ledger_account_status", sa.types.UserDefinedType(), nullable=False),
        Column("foreign_exchange_adjustment_ledger_account_id", PG_UUID, nullable=True),
        Column("cash_flow_group", sa.types.UserDefinedType(), nullable=True),
        Column("added_date", DateTime(timezone=True), nullable=False),
        Column("removed_date", DateTime(timezone=True), nullable=True),
        Column("is_used_in_journal_entries", Boolean, nullable=False),
    )

    return {
        "journal_entry": journal_entry,
        "entry_line": entry_line,
        "ledger_account": ledger_account,
    }


# ----------------- Secrets helpers -----------------

def access_secret(project_id: Optional[str], secret_name: Optional[str], version: str = "latest") -> Optional[str]:
    if not secret_name:
        return None
    client = secretmanager.SecretManagerServiceClient()
    if "/" in secret_name:
        name = secret_name
    else:
        if not project_id:
            raise ValueError("project_id is required when using secret short names")
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8")


# ----------------- Utility functions -----------------

def date_features(dt: datetime.datetime) -> Dict[str, Any]:
    d = dt.date()
    year = d.year
    month = d.month
    day = d.day
    dow = d.weekday()
    month_angle = 2.0 * math.pi * (month / 12.0)
    # Approximate day angle over 31 bucket
    day_angle = 2.0 * math.pi * (day / 31.0)
    return {
        "date_year": year,
        "date_month": month,
        "date_day": day,
        "date_dow": dow,
        "date_month_sin": math.sin(month_angle),
        "date_month_cos": math.cos(month_angle),
        "date_day_sin": math.sin(day_angle),
        "date_day_cos": math.cos(day_angle),
    }


def normalize_amounts(amounts: List[float]) -> List[float]:
    total = float(sum(amounts))
    if total <= 0.0:
        return [0.0 for _ in amounts]
    return [float(x) / total for x in amounts]


@dataclasses.dataclass
class AccountCatalog:
    id_to_index: Dict[str, int]
    rows: List[Dict[str, Any]]

    @staticmethod
    def build(conn: sa.Connection, tables: Dict[str, Table], business_id: Optional[str]) -> "AccountCatalog":
        la = tables["ledger_account"]
        stmt = sa.select(
            la.c.ledger_account_id,
            la.c.business_id,
            la.c.number,
            la.c.name,
            la.c.nature,
        )
        if business_id:
            stmt = stmt.where(la.c.business_id == business_id)
        stmt = stmt.order_by(la.c.number, la.c.name)
        rows = []
        for row in conn.execute(stmt.execution_options(stream_results=True)):
            rec = {
                "ledger_account_id": str(row.ledger_account_id),
                "business_id": row.business_id,
                "number": row.number,
                "name": row.name,
                "nature": row.nature,
            }
            rows.append(rec)
        id_to_index = {r["ledger_account_id"]: i for i, r in enumerate(rows)}
        return AccountCatalog(id_to_index=id_to_index, rows=rows)

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "generated_at": int(time.time()),
            "num_accounts": len(self.rows),
            "accounts": self.rows,
        }


# ----------------- GCS helpers -----------------

def write_gcs_json(gcs_uri: str, data: Dict[str, Any], client: storage.Client) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    _, path = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data, ensure_ascii=False, indent=2), content_type="application/json")


# ----------------- Ingestion core -----------------

def build_join_statement(
    tables: Dict[str, Table],
    business_id: Optional[str],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> sa.sql.Select:
    je = tables["journal_entry"]
    el = tables["entry_line"]
    la = tables["ledger_account"]

    stmt = (
        sa.select(
            je.c.journal_entry_id,
            je.c.business_id,
            je.c.date,
            je.c.description.label("je_description"),
            je.c.currency,
            je.c.journal_entry_type,
            je.c.journal_entry_sub_type,
            el.c.entry_line_id,
            el.c.index.label("line_index"),
            el.c.description.label("line_description"),
            el.c.debit,
            el.c.credit,
            el.c.exchange_rate,
            la.c.ledger_account_id,
            la.c.number.label("account_number"),
            la.c.name.label("account_name"),
            la.c.nature.label("account_nature"),
        )
        .select_from(
            el.join(je, el.c.journal_entry_id == je.c.journal_entry_id).outerjoin(
                la, el.c.ledger_account_id == la.c.ledger_account_id
            )
        )
    )
    conditions = []
    if business_id:
        conditions.append(je.c.business_id == business_id)
    if start_date:
        conditions.append(je.c.date >= datetime.datetime.combine(start_date, datetime.time.min, tzinfo=datetime.timezone.utc))
    if end_date:
        conditions.append(je.c.date <= datetime.datetime.combine(end_date, datetime.time.max, tzinfo=datetime.timezone.utc))
    if conditions:
        stmt = stmt.where(sa.and_(*conditions))
    stmt = stmt.order_by(je.c.date, je.c.journal_entry_id, el.c.index)
    return stmt


def write_parquet_shards(
    conn: sa.Connection,
    tables: Dict[str, Table],
    catalog: AccountCatalog,
    gcs_output_uri: str,
    shard_size: int,
    filters: Dict[str, Any],
) -> Dict[str, Any]:
    if not gcs_output_uri.startswith("gs://"):
        raise ValueError("gcs_output_uri must start with gs://")

    # Prepare output paths
    base_prefix = gcs_output_uri.rstrip("/")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    shard_index = 0
    shard_records = 0
    records_written = 0
    shard_paths: List[str] = []

    def make_shard_path(si: int) -> str:
        return f"{base_prefix}/parquet/part-{si:05d}-{timestamp}.parquet"

    current_rows: List[Dict[str, Any]] = []

    stmt = build_join_statement(
        tables=tables,
        business_id=filters.get("business_id"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
    )
    result = conn.execute(stmt.execution_options(stream_results=True))

    current_je_id: Optional[str] = None
    current_je: Dict[str, Any] = {}
    debit_accounts: List[int] = []
    credit_accounts: List[int] = []
    debit_amounts: List[float] = []
    credit_amounts: List[float] = []

    def flush_current():
        nonlocal shard_index, shard_records, records_written, current_rows
        if current_je_id is None:
            return
        row = {
            "journal_entry_id": current_je.get("journal_entry_id"),
            "business_id": current_je.get("business_id"),
            "description": current_je.get("je_description", ""),
            "currency": current_je.get("currency", ""),
            "journal_entry_type": current_je.get("journal_entry_type", ""),
        }
        row.update(date_features(current_je["date"]))
        # Targets
        deb_idxs = [int(x) for x in debit_accounts]
        cre_idxs = [int(x) for x in credit_accounts]
        row["debit_accounts"] = deb_idxs
        row["credit_accounts"] = cre_idxs
        row["debit_amounts_norm"] = normalize_amounts(debit_amounts)
        row["credit_amounts_norm"] = normalize_amounts(credit_amounts)
        current_rows.append(row)
        records_written += 1
        shard_records += 1
        if shard_records >= shard_size:
            # write shard
            out_path = make_shard_path(shard_index)
            _write_rows_parquet(current_rows, out_path)
            shard_paths.append(out_path)
            shard_index += 1
            shard_records = 0
            current_rows = []

    for r in result:
        je_id = str(r.journal_entry_id)
        if current_je_id is None:
            current_je_id = je_id
            current_je = {
                "journal_entry_id": je_id,
                "business_id": r.business_id,
                "date": r.date,
                "je_description": r.je_description or "",
                "currency": r.currency or "",
                "journal_entry_type": r.journal_entry_type or "",
                "journal_entry_sub_type": r.journal_entry_sub_type or "",
            }
        if je_id != current_je_id:
            flush_current()
            # reset
            current_je_id = je_id
            debit_accounts, credit_accounts = [], []
            debit_amounts, credit_amounts = [], []
            current_je = {
                "journal_entry_id": je_id,
                "business_id": r.business_id,
                "date": r.date,
                "je_description": r.je_description or "",
                "currency": r.currency or "",
                "journal_entry_type": r.journal_entry_type or "",
                "journal_entry_sub_type": r.journal_entry_sub_type or "",
            }

        account_id = None if r.ledger_account_id is None else str(r.ledger_account_id)
        if account_id is None:
            continue
        idx = catalog.id_to_index.get(account_id)
        if idx is None:
            continue
        debit_val = float(r.debit or 0.0)
        credit_val = float(r.credit or 0.0)
        if debit_val > 0.0:
            debit_accounts.append(int(idx))
            debit_amounts.append(debit_val)
        elif credit_val > 0.0:
            credit_accounts.append(int(idx))
            credit_amounts.append(credit_val)

    # flush last
    flush_current()
    if current_rows:
        out_path = make_shard_path(shard_index)
        _write_rows_parquet(current_rows, out_path)
        shard_paths.append(out_path)

    return {
        "generated_at": int(time.time()),
        "filters": {
            "business_id": filters.get("business_id"),
            "start_date": str(filters.get("start_date")) if filters.get("start_date") else None,
            "end_date": str(filters.get("end_date")) if filters.get("end_date") else None,
        },
        "records": records_written,
        "shards": shard_paths,
    }


def _write_rows_parquet(rows: List[Dict[str, Any]], gcs_uri: str) -> None:
    # Use pandas+pyarrow which supports list columns
    df = pd.DataFrame(rows)
    # Ensure list-like columns are objects to be handled by pyarrow
    for col in ("debit_accounts", "credit_accounts", "debit_amounts_norm", "credit_amounts_norm"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) else [])
    df.to_parquet(gcs_uri, engine="pyarrow", index=False)


def parse_date(s: Optional[str]) -> Optional[datetime.date]:
    if not s:
        return None
    return datetime.date.fromisoformat(s)


# ----------------- Main CLI -----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest from Cloud SQL and write Parquet shards to GCS.")
    parser.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--instance", type=str, default=os.environ.get("DB_INSTANCE_CONNECTION_NAME"))
    parser.add_argument("--db", type=str, default=os.environ.get("DB_NAME", "postgres"))
    parser.add_argument("--db-user", type=str, default=os.environ.get("DB_USER"))
    parser.add_argument("--db-user-secret", type=str, default=os.environ.get("DB_USER_SECRET_NAME"))
    parser.add_argument("--private-ip", action="store_true")
    parser.add_argument("--gcs-output-uri", type=str, default=os.environ.get("GCS_OUTPUT_URI"), required=False)
    parser.add_argument("--gcs-output-uri-secret", type=str, default=os.environ.get("GCS_OUTPUT_URI_SECRET_NAME"))
    parser.add_argument("--business-id", type=str, default=os.environ.get("BUSINESS_ID"))
    parser.add_argument("--start-date", type=str, default=os.environ.get("START_DATE"))
    parser.add_argument("--end-date", type=str, default=os.environ.get("END_DATE"))
    parser.add_argument("--shard-size", type=int, default=int(os.environ.get("SHARD_SIZE", "10000")))
    parser.add_argument("--secrets-project", type=str, default=os.environ.get("SECRETS_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    args = parser.parse_args()

    if not args.instance:
        parser.error("--instance (or env DB_INSTANCE_CONNECTION_NAME) is required")

    # Resolve secrets
    db_user = access_secret(args.secrets_project, args.db_user_secret) or args.db_user
    gcs_output = access_secret(args.secrets_project, args.gcs_output_uri_secret) or args.gcs_output_uri
    if not db_user:
        parser.error("--db-user (or secret) is required")
    if not gcs_output:
        parser.error("--gcs-output-uri (or secret) is required")

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    engine = create_engine_with_connector(
        instance_connection_name=args.instance,
        db_name=args.db,
        db_user=db_user,
        enable_private_ip=bool(args.private_ip),
        enable_iam_auth=True,
    )
    metadata = sa.MetaData()
    tables = reflect_or_define_tables(metadata)

    with engine.connect() as conn:
        # Build account catalog snapshot
        catalog = AccountCatalog.build(conn, tables, business_id=args.business_id)
        client = storage.Client()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        accounts_artifact_uri = f"{gcs_output.rstrip('/')}/artifacts/accounts_{timestamp}.json"
        write_gcs_json(accounts_artifact_uri, catalog.to_artifact(), client)

        manifest = write_parquet_shards(
            conn=conn,
            tables=tables,
            catalog=catalog,
            gcs_output_uri=gcs_output,
            shard_size=args.shard_size,
            filters={
                "business_id": args.business_id,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        manifest_uri = f"{gcs_output.rstrip('/')}/artifacts/manifest_{timestamp}.json"
        write_gcs_json(manifest_uri, manifest, client)
        print(json.dumps({"accounts_artifact": accounts_artifact_uri, "manifest": manifest_uri}, indent=2))


if __name__ == "__main__":
    main()


