from typing import List

import sqlite3
import pandas as pd


class SQL(object):
    NUM_PRICE_HISTORY_VALUES = 7

    def __init__(self, db_file: str):
        try:
            self.conn = sqlite3.connect(db_file)
        except Exception as e:
            print(e)

    def create_table(self, creation_sql: str):
        try:
            curs = self.conn.cursor()
            curs.execute(creation_sql)
            self.conn.commit()
        except Exception as e:
            print(e)

    def insert_pd(self, insert_sql: str, df: pd.DataFrame):
        try:
            curs = self.conn.cursor()
            data = df.to_records(index=False).tolist()
            curs.executemany(insert_sql, data)
            self.conn.commit()
        except Exception as e:
            print(e)

    @staticmethod
    def generate_create_table_sql_string(table_name: str, headers: List[str]):
        return f"""CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(headers)});"""

    @staticmethod
    def generate_insert_sql_string(table_name: str, value_count: int):
        return f"""INSERT INTO {table_name} VALUES({'?,' * (value_count-1)}?)"""


def main():
    sql = SQL(
        "/Users/ryanelliott/Documents/college/fourth_year/discrete/data/"
        "database/price_history.db"
    )
    create_table_sql = SQL.generate_create_table_sql_string(
        table_name="price_history",
        headers=[
            "security STRING NOT NULL",
            "open REAL",
            "high REAL",
            "low REAL",
            "close REAL",
            "volume REAL",
            "datetime REAL",
            "PRIMARY KEY(security, datetime)",
        ],
    )
    sql.create_table(create_table_sql)

    insert_sql = SQL.generate_insert_sql_string(
        "price_history", value_count=SQL.NUM_PRICE_HISTORY_VALUES
    )


if __name__ == "__main__":
    main()
