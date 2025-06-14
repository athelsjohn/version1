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
            timeout=7200,
            check = True
        )
        logger.info("retrain_models.py completed successfully.")
    except subprocess.TimeoutExpired:
        logger.error("Retrain process timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed with exit code: {str(e)}")
    except FileNotFoundError:
        logger.error("File not found")
    except Exception as e:
        logger.error(f"Error running retrain_models.py: {str(e)}")

# Schedule to run every 72 hours
schedule.every(72).hours.do(run_retrain)
logger.info("72-hour scheduler started. Will run retrain_models.py every 72 hours.")

while True:
    try:
        schedule.run_pending()
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        break
    except Exception as e:
        logger.info(f'Scheduler error: {str(e)}')
        break