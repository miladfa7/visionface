import os
import bz2
import logging
from pathlib import Path
from typing import Optional

import gdown
from visionface.commons.utils import get_home_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeightsDownloadError(Exception):
    """Custom exception for weights download failures."""
    pass



def get_face_models_home() -> str:
    """
    Get the home directory for storing model weights

    Returns:
        str: the home directory.
    """
    return str(os.getenv("DEEPFACE_HOME", default=os.path.expanduser("~")))


def download_model_weights(
    filename: str,
    download_url: str,
    compression_format: Optional[str] = None,
) -> Path:
    """
    Download and extract model weights from a URL.
    
    Args:
        filename: Name of the target file (without extension)
        download_url: URL to download the file from
        compression_format: File compression format ('zip', 'bz2' or None)        
    Returns:
        Path to the downloaded and extracted file
        
    Raises:
        WeightsDownloadError: If download fails
        FileNotFoundError: If home directory cannot be determined
    """
    
    home_dir = Path(get_face_models_home())

    # Create weights directory structure
    weights_dir = home_dir / ".VisionFace/weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Define target file path
    target_filepath = weights_dir / filename
    
    # Check if file already exists
    if target_filepath.exists() and target_filepath.is_file():
        logger.info(f"✓ {filename} already exists at {target_filepath}")
        return target_filepath
    
    # Download the file
    logger.info(f"Downloading {filename} model weights...")
    logger.info(f"Source URL: {download_url}")
    logger.info(f"Target directory: {weights_dir}")
    
    # Determine download filename based on compression
    if compression_format:
        download_filename = f"{filename}.{compression_format}"
        download_filepath = weights_dir / download_filename
    else:
        download_filename = filename
        download_filepath = target_filepath
    
    try:
        gdown.download(download_url, str(download_filepath), quiet=False)
        logger.info(f"✓ Successfully downloaded {download_filename}")
    except Exception as e:
        error_msg = (
            f"Failed to download {filename} from {download_url}. "
            f"Please verify the URL is accessible or download manually to {target_filepath}"
        )
        logger.error(error_msg)
        raise WeightsDownloadError(error_msg) from e
    
    # Extract file if compressed
    if compression_format:
        logger.info(f"Extracting {download_filename}...")
        _extract_compressed_file(download_filepath, target_filepath, compression_format)
        
        # Clean up compressed file after extraction
        try:
            download_filepath.unlink()
            logger.info(f"Removed compressed file: {download_filename}")
        except Exception as e:
            logger.warning(f"Could not remove compressed file {download_filename}: {e}")
    
    logger.info(f"Model weights ready at: {target_filepath}")
    return target_filepath


def _extract_compressed_file(
    compressed_filepath: Path,
    target_filepath: Path,
    compression_format: str
) -> None:
    """
    Extract a compressed file to the target location.
    
    Args:
        compressed_filepath: Path to the compressed file
        target_filepath: Path where extracted file should be saved
        compression_format: Type of compression ('bz2')
        
    Raises:
        WeightsDownloadError: If extraction fails
    """
    if compression_format.lower() == "bz2":
        try:
            with bz2.BZ2File(compressed_filepath, 'rb') as compressed_file:
                with open(target_filepath, 'wb') as target_file:
                    chunk_size = 64 * 1024  # 64KB chunks
                    while True:
                        chunk = compressed_file.read(chunk_size)
                        if not chunk:
                            break
                        target_file.write(chunk)
            
            logger.info(f"✓ Successfully extracted {compressed_filepath.name} to {target_filepath.name}")
            
        except Exception as e:
            error_msg = f"Failed to extract {compressed_filepath}: {e}"
            logger.error(error_msg)
            raise WeightsDownloadError(error_msg) from e