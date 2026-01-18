import duckdb
import os
import time

def run(ddb: duckdb.DuckDBPyConnection, sql: str, message=None) -> None:
    start = time.time()
    ddb.execute(sql)
    delta = time.time() - start
    if message:
        print(f"Done: {message} in {delta:.2f}s")

def count(ddb: duckdb.DuckDBPyConnection, table: str, message=None) -> int:
    n = ddb.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if message:
        print(f"   └─ {message}: {n:,} rows")
    else:
        print(f"   └─ Counted rows in {table}: {n:,} rows")
    return n

# Parquet is faster than csv
def save_parquet(ddb: duckdb.DuckDBPyConnection, out_dir: str, table: str, filename: str):
    path = os.path.join(out_dir, filename)
    ddb.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    print(f"   └─ wrote {filename}")


# Duckdb settings
def configure_duckdb(database_name = "mimic.duckdb", memory_limit = '8GB', temp_directory = '/mnt/d/') -> duckdb.DuckDBPyConnection:
    ddb = duckdb.connect(database_name)
    ddb.execute(f"PRAGMA memory_limit='{memory_limit}';")
    ddb.execute(f"PRAGMA temp_directory='{temp_directory}/.duckdb_tmp';")
    return ddb

def peek(ddb: duckdb.DuckDBPyConnection, table: str, n: int = 5) -> None:
    query = f"""
        SELECT *
        FROM {table}
        LIMIT {n}
        """
    ddb.query(query).show()

