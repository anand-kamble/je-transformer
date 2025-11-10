import os
import re
from contextlib import contextmanager
from urllib.parse import quote_plus

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

# Ensure environment variables from .env are loaded
load_dotenv(find_dotenv(), override=True)


class PostgresConnection:
    _engines: dict[
        str, Engine
    ] = {}  # Class variable to store engines for different databases
    _sessions: dict[
        str, scoped_session[Session]
    ] = {}  # Class variable to store session factories

    def __init__(
        self,
        database_name: str = "",
        min_connections: int = 1,
        max_connections: int = 10,
        create_if_not_exists: bool = False,
        use_migration_user: bool = False,
    ):
        """
        Initialize SQLAlchemy engine and session factory for a specific database

        Args:
            database_name: Name of the database to connect to
            min_connections: Minimum number of connections to keep in the pool
            max_connections: Maximum number of connections allowed in the pool
            create_if_not_exists: Whether to create the database if it doesn't exist
            use_migration_user: Whether to use migration user credentials
        """

        # Use environment variable for database name if not provided or empty
        if database_name == "":
            self.database_name = os.getenv("DB_NAME", "rai_dev")
        else:
            self.database_name = self._sanitize_dbname(database_name)

        self.min_connections = min_connections
        self.max_connections = max_connections

        # Use different credentials based on use_migration_user flag
        if use_migration_user:
            self.db_user = os.getenv("DB_USER_MIG", os.getenv("DB_USER", "postgres"))
            self.db_password_raw = os.getenv(
                "DB_PASSWORD_MIG", os.getenv("DB_PASSWORD", "examplepassword")
            )
        else:
            self.db_user = os.getenv("DB_USER", "postgres")
            self.db_password_raw = os.getenv("DB_PASSWORD", "examplepassword")

        # URL-encode password for use in connection URL
        self.db_password = quote_plus(self.db_password_raw)

        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", "5432"))

        if create_if_not_exists:
            self._ensure_database(self.database_name)

        self.db_url = (
            f"postgresql://{self.db_user}:"
            f"{self.db_password}@"
            f"{self.db_host}:"
            f"{self.db_port}/"
            f"{self.database_name}"
        )
        print(f"Database URL: {self.db_url}")
        self._initialize_engine()

    def _sanitize_dbname(self, raw: str) -> str:
        # allow only letters, digits, and _
        return re.sub(r"[^0-9A-Za-z_]", "_", raw.lower())

    def _initialize_engine(self) -> None:
        """Initialize the SQLAlchemy engine if it doesn't exist for this database"""
        if self.database_name not in PostgresConnection._engines:
            try:
                from sqlalchemy.engine.url import URL

                if self.db_host.startswith("/cloudsql"):
                    assert os.path.exists(self.db_host), (
                        f"Cloud SQL socket not mounted: {self.db_host}"
                    )
                    # Use Unix socket (Cloud SQL)
                    db_url = URL.create(
                        drivername="postgresql+psycopg2",
                        username=self.db_user,
                        password=self.db_password_raw,
                        database=self.database_name,
                        host=self.db_host,
                    )
                else:
                    # Use TCP connection
                    db_url = URL.create(
                        drivername="postgresql+psycopg2",
                        username=self.db_user,
                        password=self.db_password_raw,
                        database=self.database_name,
                        host=self.db_host,
                        port=self.db_port,
                    )
                engine = create_engine(
                    db_url,
                    poolclass=QueuePool,
                    pool_size=self.max_connections,
                    max_overflow=0,
                    pool_pre_ping=True,
                )
                session_factory = sessionmaker(bind=engine)
                PostgresConnection._engines[self.database_name] = engine
                PostgresConnection._sessions[self.database_name] = scoped_session(
                    session_factory
                )
            except Exception as e:
                raise Exception(f"Error creating database engine: {str(e)}")

    def _ensure_database(self, dbname: str):
        from sqlalchemy import text
        from sqlalchemy.exc import ProgrammingError

        # connect to the built-in 'postgres' database in autocommit mode
        admin_url = (
            f"postgresql://{self.db_user}:{quote_plus(self.db_password_raw)}@"
            f"{self.db_host}:{self.db_port}/postgres"
        )
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            try:
                conn.execute(
                    text(
                        f"CREATE DATABASE {dbname} "
                        "ENCODING = 'UTF8' "
                        "TEMPLATE = template0"
                    )
                )
            except ProgrammingError as e:
                # 42P04 = database already exists
                if "already exists" not in str(e):
                    raise
        admin_engine.dispose()

    @contextmanager
    def get_session(self):
        """
        Get a session using context manager

        Usage:
            with postgres.get_session() as session:
                results = session.query(Model).all()
        """
        session = PostgresConnection._sessions[self.database_name]()
        try:
            yield session
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def close_all_engines(cls) -> None:
        """Dispose all engines"""
        for engine in cls._engines.values():
            if engine:
                engine.dispose()
        cls._engines.clear()
        cls._sessions.clear()

    def close_engine(self) -> None:
        """Dispose the engine for this specific database"""
        if self.database_name in PostgresConnection._engines:
            PostgresConnection._engines[self.database_name].dispose()
            del PostgresConnection._engines[self.database_name]
            del PostgresConnection._sessions[self.database_name]

    def get_engine(self) -> Engine:
        """Return the SQLAlchemy engine for this database."""
        return PostgresConnection._engines[self.database_name]
