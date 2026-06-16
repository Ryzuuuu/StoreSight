# PROMPT: Create a Pydantic schema for visitor detection events that captures all required fields 
# for a retail store intelligence system, including entry/exit tracking, zone-based dwell time,
# staff detection, queue monitoring, and re-entry handling. Include validation for UUID v4,
# ISO-8601 timestamps, and confidence scores.
#
# CHANGES MADE: Added session_seq metadata field, adjusted queue_depth to be nullable,
# added sku_zone as a separate metadata field for product classification.

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid

class EventType(str, Enum):
    """Event types as per challenge specification"""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"

class EventMetadata(BaseModel):
    """Metadata for event enrichment"""
    queue_depth: Optional[int] = Field(None, description="Number of people in billing queue")
    sku_zone: Optional[str] = Field(None, description="Product zone label")
    session_seq: int = Field(default=1, description="Ordinal position in visitor session")

class VisitorEvent(BaseModel):
    """
    Structured behavioral event from detection pipeline.
    Matches exact schema from problem statement.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID v4 globally unique")
    store_id: str = Field(..., description="Store identifier from store_layout.json")
    camera_id: str = Field(..., description="Camera that produced event")
    visitor_id: str = Field(..., description="Re-ID token unique per session")
    event_type: EventType = Field(..., description="Event type from catalogue")
    timestamp: datetime = Field(..., description="ISO-8601 UTC from clip + frame offset")
    zone_id: Optional[str] = Field(None, description="Named zone or null for ENTRY/EXIT")
    dwell_ms: int = Field(default=0, description="Duration in milliseconds; 0 for instantaneous")
    is_staff: bool = Field(default=False, description="Staff classification flag")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence 0-1")
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    class Config:
        use_enum_values = True

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        """Ensure ISO-8601 format"""
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    def dict(self, **kwargs):
        """Override to use ISO format for timestamp"""
        d = super().dict(**kwargs)
        d['timestamp'] = self.timestamp.isoformat().replace('+00:00', 'Z')
        return d