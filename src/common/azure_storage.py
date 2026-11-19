"""Storage manager supporting Azure Blob Storage with transparent local filesystem fallback."""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.common.config import EngineConfig
from src.common.logger import get_logger

logger = get_logger("azure_storage")


class StorageManager:
    """Provides a unified interface for writing and reading Parquet datasets

    either to Azure Blob Storage or to the local data directory.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig.load()
        self.connection_string = self.config.azure_storage_connection_string
        self.container_name = self.config.azure_storage_container_name
        self.local_root = self.config.local_data_dir / "parquet"
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._blob_service_client = None

        if self.connection_string:
            try:
                from azure.storage.blob import BlobServiceClient

                self._blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                # Ensure container exists
                container_client = self._blob_service_client.get_container_client(self.container_name)
                if not container_client.exists():
                    container_client.create_container()
                logger.info(f"Connected to Azure Blob Storage container: {self.container_name}")
            except Exception as e:
                logger.warning(f"Could not connect to Azure Blob Storage ({e}). Falling back to local data directory.")
                self._blob_service_client = None
        else:
            logger.info(f"Using local filesystem storage at {self.local_root}")

    def save_parquet(self, df: pd.DataFrame, relative_path: str) -> Path:
        """Saves a DataFrame as Parquet to local storage and syncs to Azure Blob if configured.

        relative_path: e.g., 'standardized/records.parquet'
        """
        # Ensure path ends with .parquet
        if not relative_path.endswith(".parquet"):
            relative_path += ".parquet"

        local_path = self.local_root / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Write local Parquet file with pyarrow
        df.to_parquet(local_path, index=False, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {len(df):,} rows to local Parquet at {local_path}")

        # Upload to Azure Blob Storage if client is available
        if self._blob_service_client:
            try:
                blob_client = self._blob_service_client.get_blob_client(
                    container=self.container_name, blob=relative_path
                )
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.info(f"Uploaded {relative_path} to Azure Blob Storage ({self.container_name})")
            except Exception as e:
                logger.error(f"Failed to upload {relative_path} to Azure Blob Storage: {e}")

        return local_path

    def load_parquet(self, relative_path: str) -> pd.DataFrame:
        """Loads a DataFrame from Parquet, checking local storage first, then Azure Blob Storage."""
        if not relative_path.endswith(".parquet"):
            relative_path += ".parquet"

        local_path = self.local_root / relative_path

        # If not present locally but Azure client available, download it
        if not local_path.is_file() and self._blob_service_client:
            try:
                blob_client = self._blob_service_client.get_blob_client(
                    container=self.container_name, blob=relative_path
                )
                if blob_client.exists():
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(blob_client.download_blob().readall())
                    logger.info(f"Downloaded {relative_path} from Azure Blob Storage to {local_path}")
            except Exception as e:
                logger.warning(f"Could not download {relative_path} from Azure Blob: {e}")

        if not local_path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {local_path}")

        return pd.read_parquet(local_path, engine="pyarrow")

    def exists(self, relative_path: str) -> bool:
        """Checks if a Parquet dataset exists locally or in Azure Blob Storage."""
        if not relative_path.endswith(".parquet"):
            relative_path += ".parquet"
        local_path = self.local_root / relative_path
        if local_path.is_file():
            return True
        if self._blob_service_client:
            try:
                blob_client = self._blob_service_client.get_blob_client(
                    container=self.container_name, blob=relative_path
                )
                return blob_client.exists()
            except Exception:
                return False
        return False
