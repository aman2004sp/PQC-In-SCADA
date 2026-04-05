from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from threading import Thread
import time
import math
import random

store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [0]*100)
)

context = ModbusServerContext(slaves=store, single=True)

def updating_writer():
    t = 0
    while True:
        value = int(50 + 10*math.sin(t) + random.randint(-2, 2))
        context[0x00].setValues(3, 0, [value])
        print("Updated value:", value)
        t += 0.1
        time.sleep(1)

if __name__ == "__main__":
    Thread(target=updating_writer, daemon=True).start()
    print("Starting PLC...")
    StartTcpServer(context, address=("127.0.0.1", 5020))
