## Deployment
Two ESP-IDF projects that stream sensor/sample data over the network:
1. rawsender: sends raw samples as-is
2. bitsender: preprocesses and bit-packs samples to send only a few bits

Both are standard ESP-IDF CMake projects and work with:
- VS Code + [Espressif IDF extension (recommended)](https://marketplace.visualstudio.com/items?itemName=espressif.esp-idf-extension)
- ESP-IDF CLI (idf.py) from any terminal

Tested Setup:

- [Walter (4G/5G/GPS & ESP32-S3)](https://www.quickspot.io/datasheet/walter_datasheet.pdf)
- ESP-IDF v5.3 (or the version specified by the project)
- VS Code with “Espressif IDF” extension or ESP-IDF CLI
- Self-hosted CoAP Server - thanks to [libcoap](https://github.com/obgm/libcoap) 
- [HiveMQ](https://www.hivemq.com) for MQTT
- [Power Profiler 2](https://www.nordicsemi.com/Products/Development-hardware/Power-Profiler-Kit-2)

:exclamation: :exclamation: CoAP server, and account details need to be updated dependent on your setup. 

:exclamation: :exclamation: The code can not run without a microcontroller. 

Frist, deploy the program on the microcontroller, then connect to the Power Profiler. The energy required for data tranmission does depend on your connection and the available options.


