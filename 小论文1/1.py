import psutil
import os
pid=3454
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(pid).memory_info().rss / 1024 / 1024 / 1024))