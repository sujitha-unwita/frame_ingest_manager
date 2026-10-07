"""Pydantic external-message models shared by dashboard transports.

Only dashboard-originated inputs live here. Outbound events are created by
``services.event_factory`` because the AGX is their trusted producer.
"""
