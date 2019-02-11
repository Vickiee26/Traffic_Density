import time
start=time.strftime('%X')

parsed=start.split(':')

print(int(parsed[2]))