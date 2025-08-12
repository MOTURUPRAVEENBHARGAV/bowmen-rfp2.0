# logger_config.py
import logging
import os
import json
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "agent.log")

logger = logging.getLogger("agent_logger")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7, encoding="utf-8")
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def get_logger(name: Optional[str] = None):
    return logger if name is None else logger.getChild(name)

# === NEW: Narrative Work Notes Manager ===


class WorkNotesManager:
    """
    Manages and stores work notes, ensuring the agent source is included.
    """
    def __init__(self):
        self._notes: List[str] = []

    def add_note(self, agent_name: str, message: str):
        """
        Adds a new entry to the work notes, including the agent's name.
        """
        #
        # THE FIX IS HERE: Create the formatted string using the agent_name.
        #
        formatted_entry = f"[{agent_name}] {message}"
        
        # Add the complete, formatted string to the list.
        # This prevents adding the exact same note twice in a row.
        if not self._notes or self._notes[-1] != formatted_entry:
            self._notes.append(formatted_entry)
            
    def get_all_notes(self) -> List[str]:
        """Returns the complete list of formatted log strings."""
        return self._notes