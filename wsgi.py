import os
from interface.app import app
from config import config

# Set the configuration
config_name = os.getenv('FLASK_ENV', 'default')
app.config.from_object(config[config_name])

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['LOG_FILE']), exist_ok=True)

if __name__ == "__main__":
    app.run() 