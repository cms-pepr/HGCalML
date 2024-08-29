


import wandb
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import time

def is_older_than_n_months(iso_timestamp, n_months):
    # Parse the ISO timestamp
    run_start_time = datetime.fromisoformat(iso_timestamp)

    # Convert to UTC if necessary (assuming the timestamp is in the local timezone)
    run_start_time = run_start_time.replace(tzinfo=timezone.utc)

    # Get the current time in UTC
    current_time = datetime.now(timezone.utc)

    # Calculate the time difference
    difference = current_time - run_start_time

    # Calculate the cutoff date by subtracting n months from the current time
    cutoff_date = current_time - relativedelta(months=n_months)

    # Check if the run is older than n months
    return run_start_time < cutoff_date

# Replace with your wandb project and entity
project_name = 'Paper_Models'
entity_name = 'hgcalml'
specific_user='jkiesele'

# Fetch all runs for the specified project and entity
api = wandb.Api()
runs = api.runs(f"{entity_name}/{project_name}")

#invert order of runs so that first are the oldest (to delete the oldest first)
runs = list(reversed(list(runs)))


for run in runs:
    ruser = run.user
    if str(ruser)[6:-1] == specific_user:  # Replace with the specific user
        print("found run ", run.name, " by ", ruser)
        
        deleted_something = False
        for file in run.files():
            if file.name.endswith('.html'):
                print(f"Deleting file: {file.name} from run: {run.id} from run name: {run.name} started at {run.created_at}")
                file.delete()
                #wait a moment
                time.sleep(1)
                deleted_something = True
                if file in run.files():
                        print(f"Failed to delete file: {file.name} from run: {run.id}")
        
        if deleted_something:
            run.update()
            #now check again
            for file in run.files():
                if file.name.endswith('.html'):
                    print(f"Failed to delete file: {file.name} from run: {run.id}")
                        