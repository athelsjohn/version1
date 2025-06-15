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

    """Execute the update_relations.py script and handle process results.
    
    Runs the update_relations.py script as a subprocess, captures output,
    and logs process status and errors. Handles timeouts, process failures,
    missing scripts, and unexpected errors.

    Logs:
        INFO: Process start, successful completion
        ERROR: Timeout, process failure, missing script, unexpected errors
        WARNING: Process output contains error messages

    Raises:
        None: All exceptions are caught and logged
    """

    logger.info("Running update_relations.py")
    try:
        result = subprocess.run(
            ["python3", SCRIPT_PATH],
            capture_output=True,
            text=True,
            timeout=3600,
            check = True
        )
        logger.info("update_relations.py completed successfully.")
    except subprocess.TimeoutExpired:
            logger.error(f"Update process timed out at 1 hour")
    except subprocess.CalledProcessError as e:
        logger.error("Update failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        logger.error("Script not found")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    else:
         if "error" in result.stdout.lower():
              logger.warning("Process completed but output contains errors")
        
# Schedule to run every hour on the hour
schedule.every().hour.at(":00").do(run_update)
logger.info("Hourly scheduler started. Will run update_relations.py every hour.")

while True:
    try:
        schedule.run_pending()
        time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        break
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")
        break