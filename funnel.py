from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, ClassVar, Final, Iterable, final
from pyspark.sql import DataFrame, SparkSession, DataFrameWriter
from pyspark.sql.functions import current_timestamp, lit
import re


# Enum defining data loading strategies
@final
class LoadType(Enum):
    HISTORICAL: Final[auto] = auto()
    INCREMENTAL: Final[auto] = auto()


# Configuration container for Snowflake connection parameters
@dataclass(frozen=True)
class SnowflakeConfig:
    url: str
    database: str
    warehouse: str
    role: str
    schema: str
    user: str
    password: str

    # Generates Snowflake connection options dictionary
    @property
    def options(self) -> dict[str, str]:
        return {
            "sfUrl": self.url,
            "sfUser": self.user,
            "sfPassword": self.password,
            "sfDatabase": self.database,
            "sfSchema": self.schema,
            "sfWarehouse": self.warehouse,
            "sfRole": self.role,
        }


# Container for table name and path configuration
@dataclass(frozen=True)
class TableConfig:
    name: str
    path: str


# Main ETL processor handling data loading operations
@final
class DataLoader:
    DATE_PREFIX_REGEX: ClassVar[re.Pattern] = re.compile(r"^(\d{4}-\d{2}-\d{2})")
    WINDOW_SIZE: ClassVar[int] = 7

    # Metadata columns for ETL tracking
    ETL_METADATA: ClassVar[dict[str, Any]] = {
        "ETL_CREATED_DATE": current_timestamp(),
        "ETL_LAST_UPDATE_DATE": current_timestamp(),
        "CREATED_BY": lit("ETL_PROCESS"),
        "TO_PROCESS": lit(True),
        "EDW_EXTERNAL_SOURCE_SYSTEM": lit("Funnel"),
    }

    # Initializes loader with Spark session and configurations
    def __init__(
        self,
        spark: SparkSession,
        sf_config: SnowflakeConfig,
        load_type: LoadType,
        tables: Iterable[TableConfig] | None = None,
    ) -> None:
        self.spark = spark
        self.sf_config = sf_config
        self.load_type = load_type
        self.tables = tables or self._default_tables()

    # Provides default table configurations
    @classmethod
    def _default_tables(cls) -> list[TableConfig]:
        base_path = "abfss://dataarchitecture@quilitydatabricks.dfs.core.windows.net/RAW/SwitchBoardFunnel/sb_funnel_mysql_ninja_test"
        return [
            TableConfig("lead", f"{base_path}/lead"),
            TableConfig("lead_raw", f"{base_path}/lead_raw"),
            TableConfig("message", f"{base_path}/message"),
        ]

    # Generates set of dates for validation window
    def _date_window(self) -> set[str]:
        base_date = datetime.now(timezone.utc).date()
        return {
            (base_date - timedelta(days=offset)).isoformat()
            for offset in range(self.WINDOW_SIZE)
        }

    # Validates existence of required incremental files
    def _validate_incremental_paths(self, base_path: str) -> list[str] | None:
        clean_path = base_path.rstrip("/")
        try:
            files = dbutils.fs.ls(clean_path)
        except Exception:
            return None

        found_dates = set()
        for file_info in files:
            match = self.DATE_PREFIX_REGEX.match(file_info.name.split("/")[-1])
            if match:
                found_dates.add(match.group(1))

        return [clean_path] if self._date_window().issubset(found_dates) else None

    # Resolves paths based on load type
    def _get_paths(self, table: TableConfig) -> str | list[str]:
        match self.load_type:
            case LoadType.HISTORICAL:
                return table.path
            case LoadType.INCREMENTAL:
                return self._validate_incremental_paths(table.path) or []
            case _:
                raise ValueError(f"Invalid load type: {self.load_type}")

    # Loads Delta Lake data from validated paths
    def _load_data(self, path: str | list[str]) -> DataFrame:
        if isinstance(path, list) and not path:
            return self.spark.createDataFrame([], self.spark.range(0).schema)
        return (
            self.spark.read.format("delta").load(*path)
            if isinstance(path, list)
            else self.spark.read.format("delta").load(path)
        )

    # Configures Snowflake writer with load-specific options
    def _create_writer(self, df: DataFrame, table: str) -> DataFrameWriter:
        fully_qualified_name = f"{self.sf_config.database}.{self.sf_config.schema}.STG_SBFUNNEL_{table.upper()}"
        writer = (
            df.write.format("snowflake")
            .options(**self.sf_config.options)
            .option("dbtable", fully_qualified_name)
        )

        return (
            writer.mode("overwrite")
            .option("overwriteSchema", "true")
            .option("autocreateTable", "true")
            if self.load_type == LoadType.HISTORICAL
            else writer.mode("append")
            .option("column_mapping", "name")
            .option("column_mismatch_behavior", "ignore")
        )

    # Executes ETL pipeline for all configured tables
    def execute(self) -> None:
        for table in self.tables:
            paths = self._get_paths(table)
            if not paths and self.load_type == LoadType.INCREMENTAL:
                print(f"Skipping {table.name} - incomplete dataset")
                continue
            df = self._load_data(paths).withColumns(self.ETL_METADATA)
            self._create_writer(df, table.name).save()


# Main entry point for ETL execution
@final
def main() -> None:
    spark = SparkSession.builder.getOrCreate()

    sf_config = SnowflakeConfig(
        url="https://hmkovlx-nu26765.snowflakecomputing.com",
        database="DEV",
        warehouse="INTEGRATION_COMPUTE_WH",
        role="ACCOUNTADMIN",
        schema="QUILITY_EDW_STAGE",
        user=dbutils.secrets.get("key-vault-secret", "DataProduct-SF-EDW-User"),
        password=dbutils.secrets.get("key-vault-secret", "DataProduct-SF-EDW-Pass"),
    )

    DataLoader(spark, sf_config, LoadType.INCREMENTAL).execute() # Historical | Incremental


if __name__ == "__main__":
    main()
