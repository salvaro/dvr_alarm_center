"""Constants for the dvr_alarm_service integration."""

from struct import calcsize
from typing import Final

from homeassistant.components.binary_sensor import BinarySensorDeviceClass

DOMAIN = "dvr_alarm_service"

# Server defaults
DEFAULT_SERVICE_PORT: Final = 15002
DEFAULT_SERVICE_TIMEOUT: Final = 30
DEFAULT_SERVICE_BUFFER_SIZE: Final = 4096

# Server Configuration
CONF_SERVICE_PORT: Final = "service_port"
CONF_SERVICE_TIMEOUT: Final = "service_timeout"
CONF_SERVICE_BUFFER_SIZE: Final = "service_buffer_size"
CONF_SERVICE_NAME: Final = "service_name"
CONF_SERVICE_DEVICES: Final = "devices"

# Source Devices Configuration
CONF_DEVICE_ID: Final = "serial"
CONF_DEVICE_CHANNEL_COUNT: Final = "channel_count"
CONF_DEVICE_NAME: Final = "name"
CONF_DEVICE_AREA: Final = "area"
CONF_DEVICE_CHANNELS: Final = "channels"
CONF_DEVICE_SENSOR_DELAY: Final = "delay"

# Source Device Channels Configuration
CONF_CHANNELS: Final = "channels"
CONF_CHANNEL_DEVICE_ID = "device_serial"
CONF_CHANNEL_ID: Final = "channel_no"
CONF_CHANNEL_ENABLED = "enabled"
CONF_CHANNEL_NAME: Final = "name"
CONF_CHANNEL_AREA: Final = "area"

CONF_TITLE: Final = "_title"

# Sensor types
CONF_SENSOR_TYPE: Final = "sensor_type"
SENSOR_TYPE_OCCUPANCY: Final = BinarySensorDeviceClass.OCCUPANCY
SENSOR_TYPE_MOTION: Final = BinarySensorDeviceClass.MOTION
SENSOR_TYPES: Final = [SENSOR_TYPE_OCCUPANCY, SENSOR_TYPE_MOTION]

# Data storage
DATA_ALARM_SERVER: Final = "alarm_server"
DATA_CONFIG: Final = "config"
DATA_TCP_SERVER: Final = "tcp_server"

# Protocol
PACKET_HEADER_FORMAT: Final = "BB2xII2xHI"
PACKET_HEADER_SIZE: Final = calcsize(PACKET_HEADER_FORMAT)
EXPECTED_HEAD: Final = 0xFF
EXPECTED_VERSION: Final = 0x01
EXPECTED_SESSION: Final = 0x00
EXPECTED_SEQUENCE: Final = 0x00
EXPECTED_MSG_ID: Final = 1508

EVENT_SENSOR_DELAY = 10
# Events
PAYLOAD_EVENT_MAPPING = {
    # "HumanDetect": (BinarySensorDeviceClass.OCCUPANCY,),
    "HumanDetect": (BinarySensorDeviceClass.OCCUPANCY, BinarySensorDeviceClass.MOTION),
    "MotionDetect": (BinarySensorDeviceClass.MOTION,),
    "LossDetect": (BinarySensorDeviceClass.CONNECTIVITY,),
    "BlindDetect": (BinarySensorDeviceClass.PROBLEM,),
}
PAYLOAD_OCCUPANCY_EVENTS = ["HumanDetect", "MotionDetect"]
PAYLOAD_MOTION_EVENTS = ["MotionDetect"]
PAYLOAD_TYPES = ["Alarm"]
