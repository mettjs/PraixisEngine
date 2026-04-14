import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 1. Ensure the logs directory exists
Path("logs").mkdir(exist_ok=True)

# 2. Create a custom logger named "praxis"
logger = logging.getLogger("praxis")
logger.setLevel(logging.INFO) # Captures INFO, WARNING, ERROR, and CRITICAL

# 3. Define the enterprise format: [Time] | [Level] | [Message]
_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 4. The File Handler — rotates at 10 MB, keeps 5 backups before discarding
_file_handler = RotatingFileHandler(
    "logs/praxis.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
_file_handler.setFormatter(_formatter)

# 5. The Console Handler (Prints to your terminal)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)

# 6. Attach handlers (but prevent duplicates if imported multiple times)
if not logger.handlers:
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)