#!/usr/bin/env python3
"""
Data ingestion CLI: extract journal entries + entry lines from Cloud SQL (Postgres),
assemble money-flow-focused labels, and write TFRecords to GCS.

GCP-first stack:
- Cloud SQL Python Connector + SQLAlchemy (IAM auth)
- Google Cloud Storage client for artifacts (account snapshot, manifest)
- TensorFlow TFRecord writing to gs:// (requires tensorflow-io-gcs-filesystem at runtime)

This script intentionally focuses on the ingestion step only.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sqlalchemy as sa
import tensorflow as tf
from google.cloud import secretmanager, storage
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import (Boolean, Column, DateTime, Integer, Numeric, String,
                        Table, Text)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# --------------- Logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("build_tfrecords")


# --------------- DB Engine (Cloud SQL Connector + SQLAlchemy) ---------------
def create_engine_with_connector(
    instance_connection_name: str,
    db_name: str,
    db_user: str,
    enable_private_ip: bool = False,
    enable_iam_auth: bool = True,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> sa.Engine:
    """
    Create a SQLAlchemy engine using the Cloud SQL Python Connector and pg8000 driver.
    """
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


# --------------- SQLAlchemy Table Definitions (Core) ---------------
def reflect_or_define_tables(metadata: sa.MetaData) -> Dict[str, Table]:
    """
    Define tables according to Database_structure.md. We don't reflect to avoid dependency on DB perms.
    """
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


# --------------- Utilities ---------------
def ensure_gcs_client() -> storage.Client:
    return storage.Client()


def write_gcs_json(gcs_uri: str, data: Dict[str, Any], gcs_client: storage.Client) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    _, path = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path.split("/", 1)
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data, ensure_ascii=False, indent=2), content_type="application/json")


def parse_date(date_str: Optional[str]) -> Optional[datetime.date]:
    if not date_str:
        return None
    return datetime.date.fromisoformat(date_str)


def date_features(dt: datetime.datetime) -> Dict[str, Any]:
    # Convert to date in timezone-agnostic manner
    d = dt.date()
    year = d.year
    month = d.month
    day = d.day
    dow = d.weekday()  # Monday=0
    month_angle = 2.0 * math.pi * (month / 12.0)
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


def bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_list_feature(values: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_list_feature(values: List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


# --------------- Secret Manager helpers ---------------
def access_secret(project_id: Optional[str], secret_name: str, version: str = "latest") -> str:
    client = secretmanager.SecretManagerServiceClient()
    if "/" in secret_name:
        name = secret_name
    else:
        if not project_id:
            raise ValueError("project_id is required when using secret short names")
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8")


def maybe_from_secret(explicit: Optional[str], project_id: Optional[str], secret_name: Optional[str]) -> Optional[str]:
    if secret_name:
        return access_secret(project_id, secret_name)
    return explicit


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
        # inclusive lower bound
        conditions.append(je.c.date >= datetime.datetime.combine(start_date, datetime.time.min, tzinfo=datetime.timezone.utc))
    if end_date:
        # inclusive upper bound end-of-day
        conditions.append(je.c.date <= datetime.datetime.combine(end_date, datetime.time.max, tzinfo=datetime.timezone.utc))
    if conditions:
        stmt = stmt.where(sa.and_(*conditions))
    stmt = stmt.order_by(je.c.date, je.c.journal_entry_id, el.c.index)
    return stmt


def normalize_amounts(amounts: List[float]) -> List[float]:
    total = float(sum(amounts))
    if total <= 0.0:
        return [0.0 for _ in amounts]
    return [float(x) / total for x in amounts]


def write_examples_to_gcs(
    conn: sa.Connection,
    tables: Dict[str, Table],
    catalog: AccountCatalog,
    gcs_output_uri: str,
    shard_size: int,
    filters: Dict[str, Any],
) -> Dict[str, Any]:
    if not gcs_output_uri.startswith("gs://"):
        raise ValueError("gcs_output_uri must start with gs://")
    # Prepare sharded writers
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_prefix = gcs_output_uri.rstrip("/")
    records_written = 0
    shard_index = 0
    shard_records = 0
    shard_paths: List[str] = []

    def make_shard_path(si: int) -> str:
        return f"{base_prefix}/tfrecords/part-{si:05d}-{timestamp}.tfrecord"

    writer: Optional[tf.io.TFRecordWriter] = None

    def close_writer():
        nonlocal writer
        if writer is not None:
            writer.close()
            shard_paths.append(current_shard_path)

    stmt = build_join_statement(
        tables=tables,
        business_id=filters.get("business_id"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
    )
    # Stream results
    result = conn.execute(stmt.execution_options(stream_results=True))

    current_je_id: Optional[str] = None
    current_je: Dict[str, Any] = {}
    debit_accounts: List[int] = []
    credit_accounts: List[int] = []
    debit_amounts: List[float] = []
    credit_amounts: List[float] = []

    current_shard_path = make_shard_path(shard_index)
    writer = tf.io.TFRecordWriter(current_shard_path)

    for row in result:
        je_id = str(row.journal_entry_id)
        if current_je_id is None:
            # initialize first JE
            current_je_id = je_id
            current_je = {
                "journal_entry_id": je_id,
                "business_id": row.business_id,
                "date": row.date,
                "je_description": row.je_description or "",
                "currency": row.currency or "",
                "journal_entry_type": row.journal_entry_type or "",
                "journal_entry_sub_type": row.journal_entry_sub_type or "",
            }

        if je_id != current_je_id:
            # flush previous JE
            records_written, shard_records, shard_index, current_shard_path, writer = _flush_je_to_writer(
                current_je,
                debit_accounts,
                credit_accounts,
                debit_amounts,
                credit_amounts,
                catalog,
                writer,
                records_written,
                shard_records,
                shard_size,
                shard_index,
                make_shard_path,
            )
            # reset accumulators for new JE
            current_je_id = je_id
            current_je = {
                "journal_entry_id": je_id,
                "business_id": row.business_id,
                "date": row.date,
                "je_description": row.je_description or "",
                "currency": row.currency or "",
                "journal_entry_type": row.journal_entry_type or "",
                "journal_entry_sub_type": row.journal_entry_sub_type or "",
            }
            debit_accounts = []
            credit_accounts = []
            debit_amounts = []
            credit_amounts = []

        # process current line
        account_id = None if row.ledger_account_id is None else str(row.ledger_account_id)
        if account_id is None:
            continue  # skip lines without account
        account_index = catalog.id_to_index.get(account_id)
        if account_index is None:
            # Account not in snapshot (filtered out?) skip
            continue
        debit_val = float(row.debit or 0.0)
        credit_val = float(row.credit or 0.0)
        if debit_val > 0.0:
            debit_accounts.append(int(account_index))
            debit_amounts.append(debit_val)
        elif credit_val > 0.0:
            credit_accounts.append(int(account_index))
            credit_amounts.append(credit_val)
        else:
            # zero line, ignore
            pass

    # flush last JE if any
    if current_je_id is not None:
        records_written, shard_records, shard_index, current_shard_path, writer = _flush_je_to_writer(
            current_je,
            debit_accounts,
            credit_accounts,
            debit_amounts,
            credit_amounts,
            catalog,
            writer,
            records_written,
            shard_records,
            shard_size,
            shard_index,
            make_shard_path,
        )

    close_writer()

    manifest = {
        "generated_at": int(time.time()),
        "filters": {
            "business_id": filters.get("business_id"),
            "start_date": str(filters.get("start_date")) if filters.get("start_date") else None,
            "end_date": str(filters.get("end_date")) if filters.get("end_date") else None,
        },
        "records": records_written,
        "shards": shard_paths,
    }
    return manifest


def _flush_je_to_writer(
    current_je: Dict[str, Any],
    debit_accounts: List[int],
    credit_accounts: List[int],
    debit_amounts: List[float],
    credit_amounts: List[float],
    catalog: AccountCatalog,
    writer: tf.io.TFRecordWriter,
    records_written: int,
    shard_records: int,
    shard_size: int,
    shard_index: int,
    make_shard_path_fn,
) -> Tuple[int, int, int, str, tf.io.TFRecordWriter]:
    # normalize per side independently
    debit_amounts_norm = normalize_amounts(debit_amounts)
    credit_amounts_norm = normalize_amounts(credit_amounts)
    # Build TF Example
    desc_bytes = (current_je.get("je_description") or "").encode("utf-8")
    date_feats = date_features(current_je["date"])
    features = {
        "journal_entry_id": bytes_feature(current_je["journal_entry_id"].encode("utf-8")),
        "business_id": bytes_feature((current_je.get("business_id") or "").encode("utf-8")),
        "description": bytes_feature(desc_bytes),
        "currency": bytes_feature((current_je.get("currency") or "").encode("utf-8")),
        "journal_entry_type": bytes_feature((current_je.get("journal_entry_type") or "").encode("utf-8")),
        "journal_entry_sub_type": bytes_feature((current_je.get("journal_entry_sub_type") or "").encode("utf-8")),
        "date_year": tf.train.Feature(int64_list=tf.train.Int64List(value=[date_feats["date_year"]])),
        "date_month": tf.train.Feature(int64_list=tf.train.Int64List(value=[date_feats["date_month"]])),
        "date_day": tf.train.Feature(int64_list=tf.train.Int64List(value=[date_feats["date_day"]])),
        "date_dow": tf.train.Feature(int64_list=tf.train.Int64List(value=[date_feats["date_dow"]])),
        "date_month_sin": float_list_feature([date_feats["date_month_sin"]]),
        "date_month_cos": float_list_feature([date_feats["date_month_cos"]]),
        "date_day_sin": float_list_feature([date_feats["date_day_sin"]]),
        "date_day_cos": float_list_feature([date_feats["date_day_cos"]]),
        "debit_accounts": int64_list_feature(debit_accounts),
        "credit_accounts": int64_list_feature(credit_accounts),
        "debit_amounts_norm": float_list_feature(debit_amounts_norm),
        "credit_amounts_norm": float_list_feature(credit_amounts_norm),
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
    records_written += 1
    shard_records += 1
    if shard_records >= shard_size:
        # rotate shard
        writer.close()
        shard_index += 1
        shard_records = 0
        new_path = make_shard_path_fn(shard_index)
        writer = tf.io.TFRecordWriter(new_path)
        return records_written, shard_records, shard_index, new_path, writer
    # return unchanged shard path (caller holds it)
    return records_written, shard_records, shard_index, make_shard_path_fn(shard_index), writer


def main():
    parser = argparse.ArgumentParser(description="Build TFRecords from Cloud SQL journal entries.")
    parser.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--instance", type=str, default=os.environ.get("DB_INSTANCE_CONNECTION_NAME"), help="Cloud SQL instance connection name: <PROJECT>:<REGION>:<INSTANCE>")
    parser.add_argument("--db", type=str, default=os.environ.get("DB_NAME", "postgres"))
    parser.add_argument("--db-user", type=str, default=os.environ.get("DB_USER"), help="For IAM auth, set to service account email or DB user with IAM enabled.")
    parser.add_argument("--db-user-secret", type=str, default=os.environ.get("DB_USER_SECRET_NAME"), help="Secret Manager name or full resource for DB user; overrides --db-user if set.")
    parser.add_argument("--private-ip", action="store_true", help="Use Private IP to connect to Cloud SQL")
    parser.add_argument("--gcs-output-uri", type=str, default=os.environ.get("GCS_OUTPUT_URI"), required=False, help="gs://bucket/prefix for TFRecord output")
    parser.add_argument("--gcs-output-uri-secret", type=str, default=os.environ.get("GCS_OUTPUT_URI_SECRET_NAME"), help="Secret Manager name or full resource for GCS output URI; overrides --gcs-output-uri if set.")
    parser.add_argument("--business-id", type=str, default=os.environ.get("BUSINESS_ID"))
    parser.add_argument("--start-date", type=str, default=os.environ.get("START_DATE"), help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=os.environ.get("END_DATE"), help="YYYY-MM-DD")
    parser.add_argument("--shard-size", type=int, default=int(os.environ.get("SHARD_SIZE", "10000")))
    parser.add_argument("--accounts-artifacts-path", type=str, default="artifacts/accounts", help="Suffix path under gcs_output_uri for accounts JSON")
    parser.add_argument("--manifest-path", type=str, default="artifacts/manifest.json", help="Suffix path under gcs_output_uri for manifest JSON")
    parser.add_argument("--secrets-project", type=str, default=os.environ.get("SECRETS_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT"), help="Project ID for Secret Manager when using short secret names")
    args = parser.parse_args()

    # Resolve secrets/env
    resolved_db_user = maybe_from_secret(args.db_user, args.secrets_project, args.db_user_secret)
    resolved_gcs_output = maybe_from_secret(args.gcs_output_uri, args.secrets_project, args.gcs_output_uri_secret)

    if not args.instance:
        parser.error("--instance (or env DB_INSTANCE_CONNECTION_NAME) is required")
    if not resolved_db_user:
        parser.error("--db-user (or env DB_USER) is required")
    if not resolved_gcs_output:
        parser.error("--gcs-output-uri (or env GCS_OUTPUT_URI) is required")

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    logger.info("Connecting to Cloud SQL instance=%s db=%s user=%s private_ip=%s", args.instance, args.db, resolved_db_user, args.private_ip)
    engine = create_engine_with_connector(
        instance_connection_name=args.instance,
        db_name=args.db,
        db_user=resolved_db_user,
        enable_private_ip=bool(args.private_ip),
        enable_iam_auth=True,
    )
    metadata = sa.MetaData()
    tables = reflect_or_define_tables(metadata)

    with engine.connect() as conn:
        logger.info("Building account catalog snapshot (business_id=%s)", args.business_id)
        catalog = AccountCatalog.build(conn, tables, business_id=args.business_id)
        logger.info("Accounts in snapshot: %d", len(catalog.rows))

        # Write accounts artifact to GCS
        gcs_client = ensure_gcs_client()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        accounts_artifact_uri = f"{resolved_gcs_output.rstrip('/')}/{args.accounts_artifacts_path.rstrip('/')}_{timestamp}.json"
        write_gcs_json(accounts_artifact_uri, catalog.to_artifact(), gcs_client)
        logger.info("Wrote accounts artifact to %s", accounts_artifact_uri)

        # Extract and write TFRecords
        logger.info("Streaming entries and writing TFRecords to %s", resolved_gcs_output)
        manifest = write_examples_to_gcs(
            conn=conn,
            tables=tables,
            catalog=catalog,
            gcs_output_uri=resolved_gcs_output,
            shard_size=args.shard_size,
            filters={
                "business_id": args.business_id,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        # Write manifest
        manifest_uri = f"{resolved_gcs_output.rstrip('/')}/{args.manifest_path.lstrip('/')}"
        write_gcs_json(manifest_uri, manifest, gcs_client)
        logger.info("Wrote manifest to %s (records=%d shards=%d)", manifest_uri, manifest["records"], len(manifest["shards"]))


if __name__ == "__main__":
    main()


