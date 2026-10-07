"""MCU-facing service. LLD SW S3.8 step 3 calls this the "AGX Orin control
service"; S3.8a calls it "Control Service, block 7 in the diagram".

Everything in this package reads or writes the ONE RS-422 port, which is why
it is one package with one owner (mcu_link.McuLink) rather than several
independent tools.

    protocol.py             HW LLD S3.5.2 unified frame (re-export)
    transport.py            Transport ABC + SerialTransport (real port)
    mcu_link.py             THE OWNER -- one reader, dispatches to the rest
    heartbeat_receiver.py   HW LLD S3.5.2 beats; LLD SW S3.8a link state
    telemetry_receiver.py   LLD SW S3.6 trigger continuity
    mcu_command.py          LLD SW S3.8a command retry + ACK correlation
    mcu_state.py            OBSERVES HW LLD S3.5.3 firmware state machine
"""
