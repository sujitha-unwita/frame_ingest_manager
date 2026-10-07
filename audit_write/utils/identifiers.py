"""Generate database identities independently of MCU sequence numbers."""

from uuid import UUID, uuid4


class Identifiers:
    """One UUID generation point for audit business operations."""

    @staticmethod
    def new_id() -> UUID:
        """Return a new UUID for a roll, trigger, media, or raw detection."""
        return uuid4()
