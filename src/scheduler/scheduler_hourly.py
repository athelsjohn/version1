import schedule
import time
import subprocess
import logging
from logging.handlers import RotatingFileHandler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SCHEDULER_HOURLY: %(message)s",
    handlers=[
        RotatingFileHandler("scheduler_hourly.log", maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SCRIPT_PATH = "/home/athel/Desktop/Litmus7/order/update_relations.py"

def run_update():
    logger.info("Running update_relations.py")
    try:
        result = subprocess.run(
            ["python3", SCRIPT_PATH],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        if result.returncode == 0:
            logger.info("update_relations.py completed successfully.")
            logger.debug(result.stdout)
        else:
            logger.error(f"update_relations.py failed: {result.stderr}")
    except Exception as e:
        logger.exception("Error running update_relations.py")

# Schedule to run every hour on the hour
schedule.every().hour.at(":00").do(run_update)
logger.info("Hourly scheduler started. Will run update_relations.py every hour.")

while True:
    schedule.run_pending()
    time.sleep(60)
