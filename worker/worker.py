import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'opt_app'))
from time import sleep
# from app.models import DATABASE_URL
from models import CalculationTask

def run():
    while True:
        # current_calc_task = CalculationTask.query.first()
        print('No jobs, sleeping for 1 minute')
        sleep(60)
        continue


if __name__ == '__main__':
    run()
