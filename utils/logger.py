import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    """Custom logger for the AI detection system."""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (default: INFO)
        """
        self.log_dir = log_dir
        self.log_level = log_level
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('AInterview')
        self.logger.setLevel(log_level)
        
        # Create formatters and handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up file and console handlers with formatters."""
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler
        log_file = os.path.join(
            self.log_dir,
            f'ainterview_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.log_level)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_analysis_start(self, file_id: str):
        """Log the start of an analysis session."""
        self.logger.info(f"Starting analysis for file: {file_id}")

    def log_analysis_complete(self, file_id: str, confidence_score: float):
        """Log the completion of an analysis session."""
        self.logger.info(
            f"Analysis complete for file {file_id}. "
            f"Confidence score: {confidence_score:.2%}"
        )

    def log_feature_scores(self, file_id: str, features: dict):
        """Log individual feature scores."""
        self.logger.debug(f"Feature scores for {file_id}:")
        for feature, score in features.items():
            self.logger.debug(f"- {feature}: {score:.2%}")

    def log_error(self, error_msg: str, exc: Optional[Exception] = None):
        """Log an error with optional exception details."""
        if exc:
            self.logger.error(f"{error_msg}: {str(exc)}", exc_info=True)
        else:
            self.logger.error(error_msg)

    def log_warning(self, warning_msg: str):
        """Log a warning message."""
        self.logger.warning(warning_msg)

    def log_processing_step(self, step_name: str, file_id: str):
        """Log a processing step."""
        self.logger.info(f"Processing step '{step_name}' for file {file_id}")

    def log_model_load(self, model_name: str):
        """Log model loading events."""
        self.logger.info(f"Loading model: {model_name}")

    def log_file_operation(self, operation: str, file_path: str, success: bool):
        """Log file operations."""
        status = "successful" if success else "failed"
        self.logger.info(f"File {operation} {status}: {file_path}")

# Create a default logger instance
default_logger = Logger()
