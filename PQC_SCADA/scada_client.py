from pymodbus.client.sync import ModbusTcpClient
import time

client = ModbusTcpClient("127.0.0.1", port=5020)

if not client.connect():
    print("❌ Failed to connect")
    exit()

print("✅ SCADA connected")

while True:
    result = client.read_holding_registers(0, 1)

    if result.isError():
        print("❌ Error reading from PLC")
    else:
        print("📊 Value:", result.registers[0])

    time.sleep(1)
