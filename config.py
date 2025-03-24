import os
from pathlib import Path

class Config:
    """Base configuration."""
    # Base directory
    BASE_DIR = Path(__file__).resolve().parent

    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # File storage settings
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}
    
    # API Keys
    GLADIA_API_KEY = os.getenv('GLADIA_API_KEY', '')
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'logs' / 'app.log'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    # Use temporary directories for testing
    UPLOAD_FOLDER = Path('/tmp/test_uploads')
    LOG_FILE = Path('/tmp/test.log')

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # In production, these should be set through environment variables
    SECRET_KEY = os.getenv('SECRET_KEY')
    GLADIA_API_KEY = os.getenv('GLADIA_API_KEY')
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    LOG_FILE = '/var/log/ainterview/app.log'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 