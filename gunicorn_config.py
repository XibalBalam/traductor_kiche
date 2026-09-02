import multiprocessing

# Gunicorn configuration file for production deployment
# Binds to port 5001 (or another of your choosing)
bind = "0.0.0.0:5001"

# The number of worker processes for handling requests.
# Since ML models use significant memory, start with a low number of workers to prevent OOM errors.
workers = 1

# Threads per worker
threads = 2

# Timeout for workers (important for long-running inference tasks)
timeout = 120

# Reload on code changes (set to False in production)
reload = False

# Application module to run
wsgi_app = "app:app"

def on_starting(server):
    # This function is executed before the master process is initialized.
    # We can do our S3 sync here for production.
    from app import sync_training_data_from_s3
    print("Starting Gunicorn... Syncing data from S3...")
    sync_training_data_from_s3()
