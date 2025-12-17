# /workspace/Project/celery_config.py
import logging
import os
from celery import Celery
from celery.signals import setup_logging, worker_ready

# Clear the log file on startup
def clear_celery_log():
    """Clear the celery log file to prevent it from becoming too large."""
    log_file = '/workspace/Project/celery.log'
    try:
        if os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')  # Clear the file
            print(f"Cleared celery log file: {log_file}")
    except Exception as e:
        print(f"Error clearing celery log file: {e}")

# Clear log file immediately when module is loaded
clear_celery_log()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/Project/celery.log'),
        logging.StreamHandler()  # Keep terminal output
    ]
)

# Configure Celery’s logger and sub-loggers
for logger_name in ['celery', 'celery.worker', 'celery.task', 'celery.app', 'celery.backends']:
    logger = logging.getLogger(logger_name)
    logger.handlers = []  # Clear default handlers
    logger.addHandler(logging.FileHandler('/workspace/Project/celery.log'))
    logger.addHandler(logging.StreamHandler())  # Keep terminal output
    logger.setLevel(logging.DEBUG)

# Signal to prevent Celery from overriding logging
@setup_logging.connect
def configure_logging(**kwargs):
    pass  # Prevent Celery’s default logging setup

# Define the Celery instance
celery = Celery(
    'project',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure Celery settings
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    include=['tasks']
)

logging.info("Celery app initialized with tasks: %s", celery.conf.include)

# Function to initialize Celery with Flask app context
def init_celery(app):
    celery.conf.update(
        broker_url=app.config['CELERY_BROKER_URL'],
        result_backend=app.config['result_backend']
    )
    # Attach Flask app context to tasks
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask
    logging.info("Celery initialized with Flask app")

# Log registered tasks for debugging
@celery.on_after_configure.connect
def log_tasks(sender, **kwargs):
    logging.info("Registered tasks: %s", list(sender.tasks.keys()))

# Clear log when worker is ready
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Clear log file when worker starts up."""
    clear_celery_log()
    logging.info("Celery worker ready and log file cleared")