import sqlite3
from typing import Any, List, Union

import pandas as pd

from discrete.config import price_history_sql_path


class StockSQL(object):
    NUM_PRICE_HISTORY_VALUES = 7
    PRICE_HISTORY_COLS = [
        "security",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "datetime",
    ]

    def __init__(self, db_file: str):
        try:
            self.conn = sqlite3.connect(db_file)
        except Exception as e:
            print(e)

    def create_table(self, sql: str):
        try:
            curs = self.conn.cursor()
            curs.execute(sql)
            self.conn.commit()
        except Exception as e:
            print(e)

    def insert_pd(self, sql: str, df: pd.DataFrame):
        try:
            curs = self.conn.cursor()
            data = df.to_records(index=False).tolist()
            curs.executemany(sql, data)
            self.conn.commit()
        except Exception as e:
            print(e)

    def select(self, sql: str):
        try:
            curs = self.conn.cursor()
            curs.execute(sql)
            return curs.fetchall()
        except Exception as e:
            print(e)

    def select_cols(
            self, table_name: str, cols: List[str], where: str = None
    ) -> List[Any]:
        columns = ",".join(cols)
        sql = StockSQL.generate_select_sql_string(
            table_name, columns, where=where
        )
        print(sql)
        return self.select(sql)

    def fetch_securities(
            self,
            securities: Union[List[str], str],
            meta: str,
            table: str
    ) -> pd.DataFrame:
        if isinstance(securities, str):
            securities = [securities]
        comp = StockSQL.parse_meta(meta)
        for sec in securities:
            yield pd.DataFrame(
                self.select_cols(
                    table, comp, where=f"security='{sec.upper()}'"
                ), columns=comp
            )

    @staticmethod
    def parse_meta(meta: str) -> List[str]:
        comp = []
        if "O" in meta:
            comp.append("open")
        if "H" in meta:
            comp.append("high")
        if "L" in meta:
            comp.append("low")
        if "C" in meta:
            comp.append("close")
        if "V" in meta:
            comp.append("volume")
        comp.append("datetime")
        return comp

    @staticmethod
    def generate_create_table_sql_string(
            table_name: str,
            headers: List[str]
    ) -> str:
        return f"""CREATE TABLE IF NOT EXISTS {table_name} (
{", ".join(headers)});"""

    @staticmethod
    def generate_insert_sql_string(table_name: str, value_count: int) -> str:
        return f"""INSERT INTO {table_name} VALUES(
{'?,' * (value_count - 1)}?)"""

    @staticmethod
    def generate_select_sql_string(
            table_name: str, columns: str, where: str = None
    ) -> str:
        if where is None:
            return f"""SELECT {columns} FROM {table_name}"""
        return f"""SELECT {columns} FROM {table_name} WHERE {where}"""


def main():
    sql = StockSQL(price_history_sql_path)
    create_table_sql = StockSQL.generate_create_table_sql_string(
        table_name="price_history",
        headers=[
            "security STRING NOT NULL",
            "open REAL",
            "high REAL",
            "low REAL",
            "close REAL",
            "volume REAL",
            "datetime INTEGER",
            "PRIMARY KEY(security, datetime)",
        ],
    )
    sql.create_table(create_table_sql)


if __name__ == "__main__":
    main()
