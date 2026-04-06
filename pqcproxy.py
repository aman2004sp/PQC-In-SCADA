import oqs
import time
import threading
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext

PLC_IP = "127.0.0.1"
PLC_PORT = 502

PROXY_IP = "0.0.0.0"
PROXY_PORT = 5020

plc = ModbusTcpClient(PLC_IP, port=PLC_PORT)
if not plc.connect():
    print("Cannot connect to OpenPLC")
    exit()

store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [0]*100)
)
context = ModbusServerContext(slaves=store, single=True)

def proxy_loop():
    print("PQC Proxy loop started...")

    while True:
        try:
            start = time.perf_counter()

            result = plc.read_holding_registers(0, 1)
            if result.isError():
                print("PLC read error")
                time.sleep(1)
                continue

            value = result.registers[0]

            with oqs.KeyEncapsulation("ML-KEM-512") as kem:
                pk = kem.generate_keypair()
                ct, ss1 = kem.encap_secret(pk)
                ss2 = kem.decap_secret(ct)

            context[0x00].setValues(3, 0, [value])

            latency = (time.perf_counter() - start) * 1000

            print(f"Value: {value} | Latency: {latency:.3f} ms")

            time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    threading.Thread(target=proxy_loop, daemon=True).start()

    print(f"Starting PQC Proxy on {PROXY_IP}:{PROXY_PORT}")

    StartTcpServer(context, address=(PROXY_IP, PROXY_PORT))
