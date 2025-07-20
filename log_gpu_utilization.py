import time
import csv
import threading
from pathlib import Path
from datetime import datetime
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo

def log_gpu_utilization(interval, log_file: Path, stop_flag=None, new_thread=False):
    if new_thread:
        thread = threading.Thread(target=log_gpu_utilization, kwargs={
            'interval': interval,
            'log_file': log_file,
            'stop_flag': stop_flag,
            'new_thread': False
        })
        thread.start()
        return thread

    nvmlInit()
    device_count = 1  # Set this based on your job; use nvmlDeviceGetCount() if needed

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gpu_index', 'gpu_util%', 'mem_used_MB', 'mem_total_MB'])

        while not stop_flag() if stop_flag else True:
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                writer.writerow([
                    datetime.now().isoformat(),
                    i,
                    util.gpu,
                    mem.used // (1024 ** 2),
                    mem.total // (1024 ** 2)
                ])
            f.flush()
            time.sleep(interval)

    nvmlShutdown()


if __name__ == "__main__":
    log_gpu_utilization(interval=10, log_file=Path('gpu_utilization.csv'))
