import schedule
import time
import subprocess
import logging
from logging.handlers import RotatingFileHandler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SCHEDULER_RETRAIN: %(message)s",
    handlers=[
        RotatingFileHandler("scheduler_retrain.log", maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SCRIPT_PATH = "/home/athel/Desktop/Litmus7/order/retrain_models.py"

def run_retrain():
    logger.info("Running retrain_models.py")
    try:
        result = subprocess.run(
            ["python3", SCRIPT_PATH],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        if result.returncode == 0:
            logger.info("retrain_models.py completed successfully.")
            logger.debug(result.stdout)
        else:
            logger.error(f"retrain_models.py failed: {result.stderr}")
    except Exception as e:
        logger.exception("Error running retrain_models.py")

# Schedule to run every 72 hours
schedule.every(72).hours.do(run_retrain)
logger.info("72-hour scheduler started. Will run retrain_models.py every 72 hours.")

while True:
    schedule.run_pending()
    time.sleep(60)
