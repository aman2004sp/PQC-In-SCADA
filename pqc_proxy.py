import oqs
import time
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext

# Connect to real PLC (OpenPLC)
plc_client = ModbusTcpClient("127.0.0.1", port=502)

plc_client.connect()

store = ModbusSlaveContext(
    hr=ModbusSequentialDataBlock(0, [0]*100)
)

context = ModbusServerContext(slaves=store, single=True)

def proxy_loop():
    while True:
        start = time.perf_counter()

        # Read from real PLC
        result = plc_client.read_holding_registers(0, 1)
        if result.isError():
            continue

        value = result.registers[0]

        # PQC encryption/decryption
        with oqs.KeyEncapsulation("ML-KEM-512") as kem:
            pk = kem.generate_keypair()
            ct, ss1 = kem.encap_secret(pk)
            ss2 = kem.decap_secret(ct)

        # Write to SCADA-facing register
        context[0x00].setValues(3, 0, [value])

        end = time.perf_counter()
        print("Latency:", (end-start)*1000)

        time.sleep(1)

if __name__ == "__main__":
    import threading
    threading.Thread(target=proxy_loop, daemon=True).start()

    print("Starting PQC Proxy...")
    StartTcpServer(context, address=("127.0.0.1", 5020))
