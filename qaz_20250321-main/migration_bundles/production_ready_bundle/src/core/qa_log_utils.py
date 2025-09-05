
import datetime

def write_log(message, logfile='qa_log.txt'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
