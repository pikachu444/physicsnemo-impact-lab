"""Stdlib Windows resource gate for short CPU-only postprocessing workers."""
import ctypes
from ctypes import wintypes
import json
import time

class MemoryStatus(ctypes.Structure):
    _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD),*[(name,ctypes.c_ulonglong) for name in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]]
class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),*[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]]
def available():
    record=MemoryStatus();record.dwLength=ctypes.sizeof(record)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(record)):raise ctypes.WinError()
    return int(record.ullAvailPhys)
def memory():
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=ctypes.c_void_p
    counter=Counters();counter.cb=ctypes.sizeof(counter)
    function=ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),wintypes.DWORD]
    if not function(process(),ctypes.byref(counter),counter.cb):raise ctypes.WinError()
    return int(counter.WorkingSetSize),int(counter.PeakWorkingSetSize)
def guard(stage,minimum_available=1024**3,maximum_rss=200*1024**2):
    wait_started=time.monotonic();next_notice=0
    while available()<minimum_available:
        elapsed=time.monotonic()-wait_started
        if elapsed>=next_notice:
            print(json.dumps({'status':'waiting_for_1GiB_available','stage':stage,'available_bytes':available(),'wait_seconds':elapsed}),flush=True)
            next_notice=elapsed+30
        time.sleep(2)
    current,peak=memory()
    if peak>maximum_rss:raise RuntimeError(f'Own process exceeded200MiB RSS: {current}/{peak} bytes during{stage}')
    return {'available_bytes':available(),'working_set_bytes':current,'peak_working_set_bytes':peak}
