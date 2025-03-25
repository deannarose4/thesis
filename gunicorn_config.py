# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # Use only one worker to minimize memory usage
worker_class = 'sync'
worker_connections = 1000
timeout = 120  # Increase timeout to 120 seconds
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'waste-classifier'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None 