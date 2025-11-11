#!/usr/bin/env python3
"""
Simple function to test database connection using PostgresConnection.

Usage:
    python test_db_connection.py
"""

import os
import sys
from sqlalchemy import text

# Add the project root to the path if needed
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data.psql_connection import PostgresConnection  # noqa: E402


def test_db_connection(database_name: str = "", verbose: bool = True) -> bool:
    postgres = None
    try:
        if verbose:
            print("Testing database connection...")
            print(f"Database name: {database_name or 'default from .env'}")
        postgres = PostgresConnection(database_name=database_name)
        if verbose:
            print("✓ Connection object created successfully")
            print(f"  Host: {postgres.db_host}")
            print(f"  Port: {postgres.db_port}")
            print(f"  User: {postgres.db_user}")
            print(f"  Database: {postgres.database_name}")
        with postgres.get_session() as session:
            result = session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            if row and row[0] == 1:
                if verbose:
                    print("✓ Database connection test successful!")
                    print(f"  Query result: {row[0]}")
                return True
            else:
                if verbose:
                    print("✗ Database connection test failed: Unexpected query result")
                return False
    except Exception as e:
        if verbose:
            print(f"✗ Database connection test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        return False
    finally:
        if postgres is not None:
            try:
                postgres.close_engine()
                if verbose:
                    print("✓ Connection closed")
            except Exception:
                pass


if __name__ == "__main__":
    # Allow database name to be passed as command line argument
    db_name = sys.argv[1] if len(sys.argv) > 1 else ""
    
    success = test_db_connection(database_name=db_name, verbose=True)
    sys.exit(0 if success else 1)

