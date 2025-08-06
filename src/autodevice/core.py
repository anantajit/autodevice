import numpy as np

cuda = True

try:
    from pynvml import *
    from pprint import pprint
except:
    cuda = False

COLUMNS = {
    "mem_usage": 0,
    "mem_capacity": 1, 
    "mem_free": 2,
    "process_count": 3
}

def init():
    nvmlInit()
def shutdown():
    nvmlShutdown()

def get_devices(units=1e3):
    init()

    deviceCount = nvmlDeviceGetCount()
    devices = np.zeros((deviceCount, len(COLUMNS)))

    for device in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(device)

        meminfo = nvmlDeviceGetMemoryInfo(handle)
        processes = nvmlDeviceGetComputeRunningProcesses(handle)

        devices[device][COLUMNS["mem_usage"]] = meminfo.used/units
        devices[device][COLUMNS["mem_capacity"]] = meminfo.total/units
        devices[device][COLUMNS["mem_free"]] = meminfo.free/units
        devices[device][COLUMNS["process_count"]] = len(processes)

    shutdown()
    return devices
