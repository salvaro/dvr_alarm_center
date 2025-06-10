"""TCP Server for DVR Alarm Server."""

import asyncio
from collections.abc import Callable
import contextlib
from datetime import UTC, datetime, timedelta
import ipaddress
from ipaddress import IPv4Address
import json
import logging
import struct
from types import CoroutineType
from typing import Any, Literal, Union, override

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry, ConfigSubentryData
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_CHANNEL_AREA,
    CONF_CHANNEL_ENABLED,
    CONF_CHANNEL_ID,
    CONF_CHANNEL_NAME,
    CONF_DEVICE_AREA,
    CONF_DEVICE_CHANNEL_COUNT,
    CONF_DEVICE_CHANNELS,
    CONF_DEVICE_ID,
    CONF_DEVICE_NAME,
    CONF_DEVICE_SENSOR_DELAY,
    CONF_SERVICE_BUFFER_SIZE,
    CONF_SERVICE_NAME,
    CONF_SERVICE_PORT,
    CONF_SERVICE_TIMEOUT,
    DATA_ALARM_SERVER,
    DEFAULT_SERVICE_BUFFER_SIZE,
    DEFAULT_SERVICE_PORT,
    DEFAULT_SERVICE_TIMEOUT,
    DOMAIN,
    EXPECTED_HEAD,
    EXPECTED_MSG_ID,
    EXPECTED_SEQUENCE,
    EXPECTED_SESSION,
    EXPECTED_VERSION,
    PACKET_HEADER_FORMAT,
    PACKET_HEADER_SIZE,
    PAYLOAD_EVENT_MAPPING,
)

_LOGGER = logging.getLogger(__package__)


@callback
def _packet_callback(hass: HomeAssistant, service: "Service", payload: dict[str, Any]):
    """Handle incoming TCP packet payload."""
    _LOGGER.debug("Packet payload: %s", payload)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,  # AddEntitiesCallback,
) -> None:
    """Set up binary sensors from config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(entry.entry_id, {})
    hass.data[DOMAIN][entry.entry_id].setdefault(DATA_ALARM_SERVER, None)
    hass.data[DOMAIN][entry.entry_id].setdefault("entities", {})

    service: Service | None = hass.data[DOMAIN][entry.entry_id].get(DATA_ALARM_SERVER)
    hass.data[DOMAIN][entry.entry_id].setdefault("entities", {})
    if service is None:
        service = Service(hass, entry, None)
    else:
        await service.async_stop()
        service = Service(hass, entry, None)
    hass.data[DOMAIN][DATA_ALARM_SERVER] = service
    device_registry = dr.async_get(hass)
    for registry in service.get_device_registries():
        device_registry.async_get_or_create(**registry)
    for subentry_id, entities in service.get_entities(include_state=True).items():
        async_add_entities(entities, config_subentry_id=subentry_id)
    await service.async_start()


async def async_unload_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> bool:
    """Handle removal of an entry."""
    service: Service | None = hass.data[DOMAIN][entry.entry_id].get(DATA_ALARM_SERVER)
    if service:
        await service.async_stop()
        hass.data[DOMAIN][entry.entry_id].pop(DATA_ALARM_SERVER)
        entities = hass.data[DOMAIN][entry.entry_id].get("entities", {})
        for entity in entities:
            entities.pop(entity)

    entities = hass.data[DOMAIN][entry.entry_id].get["entities"]
    # for entity in entities:
    #     await entity.async_remove()  # or hass.states.async_remove(entity.entity_id)
    return True


class Device:
    """Representation of a device."""

    _service: "Service"
    _subentry_id: str
    _serial: str
    _channel_count: int
    _delay: int
    _name: str | None
    _area: str | None
    _channels: dict[int, "Channel"]

    def __init__(
        self,
        service: "Service",
        config_subentry: ConfigSubentry | None = None,
    ) -> None:
        """Create a device."""
        self._service: Service = service
        self.hass = service.hass
        self._subentry_id: str = config_subentry.subentry_id
        device_config: ConfigSubentryData = config_subentry.data
        self._serial: str = device_config.get(CONF_DEVICE_ID)
        self._channel_count: int = device_config.get(CONF_DEVICE_CHANNEL_COUNT, 1)
        self._name: str | None = (
            device_config.get(CONF_DEVICE_NAME)
            or f"Device {self._serial} on port {self._service.port}"
        )
        self._area: str | None = device_config.get(CONF_DEVICE_AREA)
        self._delay: int = device_config.get(CONF_DEVICE_SENSOR_DELAY, 0)

        self._unique_id: str = f"{self.service._unique_id}_{self._subentry_id}"  # noqa: SLF001
        self._key: str = f"{self.service._key}:{self.serial}"  # noqa: SLF001

        self._channels: dict[int, Channel] = {}

        for channel_config in device_config.get(CONF_DEVICE_CHANNELS, {}).values():
            self.add_channel(Channel(self, channel_config))

        super().__init__()

    @property
    def _path(self) -> tuple[int, str]:
        return (
            self._service.port,
            self._serial,
        )

    @property
    def service(self) -> "Service":
        """Return parent service instance."""
        return self._service

    @property
    def serial(self) -> str:
        """Return the serial number of the device."""
        return self._serial

    @property
    def channel_count(self) -> int:
        """Return the number of channels for the device."""
        return self.channel_count

    @property
    def delay(self) -> int:
        """Return the event sensors ON/OFF delay seconds for the device channels."""
        return self._delay

    @property
    def channels(self) -> dict[str, "Channel"]:
        """Return the channels dict of the device."""
        return self._channels

    @property
    def _entry_id(self) -> str:
        """Return config entry id."""
        return self._service._entry_id  # noqa: SLF001

    def add_channel(self, channel: "Channel") -> "Channel":
        """Add a channel to the device."""
        if channel.channel_no not in self.channels:
            self._channels[channel.channel_no] = channel
        return self._channels[channel.channel_no]

    def setup_channel(
        self,
        channel_no: int,
        enabled: bool = True,
        name: str | None = None,
        area: str | None = None,
    ) -> int | None:
        """Configure a device channel."""
        if channel_no > self._channel_count:
            return None
        channel = Channel(self, channel_no, enabled, name, area)
        return self.add_channel(channel)

    def _get_device_registry(self) -> dict[str, Any]:
        """Return device registry keyword arguments."""
        return {
            "config_entry_id": self._entry_id,
            "identifiers": {(DOMAIN, self._unique_id)},
            "manufacturer": "DVR Alarm Service",
            "model": "DVR Alarm Device",
            "suggested_area": self._area,
            "name": self._name,
            "serial_number": f"{self._serial}",
            "entry_type": dr.DeviceEntryType.SERVICE,
            "via_device": (DOMAIN, self._service._unique_id),  # noqa: SLF001
        }


class Channel:
    """Representation of a channel."""

    _device: Device
    _channel_no: int
    _enabled: bool
    _name: str | None
    _area: str | None
    _unique_id: str
    _key: str
    _sensors: dict[str, "EventSensor"]

    def __init__(
        self,
        device: "Device",
        channel_config: dict[str | int, Any] | None = None,
    ) -> None:
        """Create a channel."""
        self._device: Device = device
        self.hass = self.device.hass
        self._channel_no: int = channel_config.get(CONF_CHANNEL_ID, 1)
        self._enabled: bool = channel_config.get(CONF_CHANNEL_ENABLED, True)
        self._name: str | None = (
            channel_config.get(CONF_CHANNEL_NAME)
            or f"{self.device._name} Channel #{self._channel_no}"  # noqa: SLF001
        )
        self._area: str | None = channel_config.get(CONF_CHANNEL_AREA)

        self._unique_id: str = f"{self.device._unique_id}_{self._channel_no}"  # noqa: SLF001
        self._key: str = f"{self.device._key}:{self.channel_no:02d}"  # noqa: SLF001

        event_types: set[BinarySensorDeviceClass] = set()
        self._sensors: dict[str, EventSensor] = {}
        for values in PAYLOAD_EVENT_MAPPING.values():
            event_types.update(values)
        for event_type in event_types:
            inverse = event_type is BinarySensorDeviceClass.CONNECTIVITY
            event_sensor = EventSensor(self, event_type, self.delay, inverse)
            self._sensors[event_sensor.event_type] = event_sensor
        super().__init__()

    @property
    def _path(self) -> tuple[int, str, int]:
        return (self.service.port, self.device.serial, self.channel_no)

    @property
    def channel_no(self) -> int:
        """Return the channel number."""
        return self._channel_no

    @property
    def service(self) -> "Service":
        """Return channel device service."""
        return self.device.service

    @property
    def device(self) -> "Device":
        """Return channel device."""
        return self._device

    @property
    def serial(self) -> str:
        """Return device serial."""
        return self.device.serial

    @property
    def delay(self) -> str:
        """Return  event sensors ON/OFF delay seconds for the channel."""
        return self.device.delay

    @property
    def sensors(self) -> dict[BinarySensorDeviceClass, "EventSensor"]:
        """Return the sensors dictionary for the channel."""
        return self._sensors

    @property
    def _entry_id(self) -> str:
        return self.device._entry_id  # noqa: SLF001

    @property
    def _subentry_id(self) -> str:
        return self.device._subentry_id  # noqa: SLF001


class DeduplicationHelper:
    """Represent a class to help with binary state deduplication."""

    __slots__ = ("_last",)

    def __init__(self) -> None:
        """Create the deduplication helper."""
        self._last = None

    def next(self, value):
        """Check if the value is different from the last one."""
        if self._last == value:
            return False
        self._last = value
        return True

    def __repr__(self):
        """Return the deduplication helper class string representation."""
        return f"{self.__class__.__name__}()"


class Filter:
    """Represent the base class for filters."""

    __slots__ = (
        "_dedup",
        "_parent",
        "_task_creator",
        "_tasks",
        "next",
    )

    ON = Literal["ON"]
    OFF = Literal["OFF"]
    ON_OFF = Literal["ON/OFF"]
    TIMING = Literal["TIMING"]
    SETTLE = Literal["SETTLE"]

    def __init__(self) -> None:
        """Create the filter."""
        self._dedup = DeduplicationHelper()
        self.next: Filter | None = None
        self._parent: EventSensor | None = None
        self._tasks: dict[type[str], asyncio.Task] = {}
        self._task_creator = asyncio.create_task

    async def async_input(self, value: bool, is_initial: bool):
        """Input pipe."""
        result = await self.async_new_value(value, is_initial)
        if result is not None:
            await self.async_output(result, is_initial)

    async def async_output(self, value: bool, is_initial: bool):
        """Output pipe."""
        if not self._dedup.next(value):
            return
        if self.next is None:
            await self._parent._async_send_state_internal(value, is_initial)  # noqa: SLF001
        else:
            await self.next.async_input(value, is_initial)

    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        raise NotImplementedError

    async def async_cleanup(self):
        """Cancel the tasks and clean up."""
        for name, task in self._tasks.items():  # noqa: B007, PERF102
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tasks = {}

    def _set_timeout(self, name: type[str], delay: float, callback_: Callable):
        """Cancel any existing name task and crete one."""
        self._cancel_timeout(name)
        self._tasks[name] = self._task_creator(
            self._async_timeout_task(name, delay, callback_)
        )

    def _cancel_timeout(self, name: type[str]):
        """Cancel any existing named task."""
        task = self._tasks.pop(name, None)
        if task and not task.done():
            task.cancel()

    @staticmethod
    async def _async_timeout_task(name: type[str], delay: float, callback_: Callable):
        """Implement the timeout task to execute the given callback after the delay."""
        try:
            await asyncio.sleep(delay)
            await callback_()
        except asyncio.CancelledError:
            pass

    def __repr__(self) -> str:
        """Return the filter class string representation."""
        raise NotImplementedError


class DelayedOnOffFilter(Filter):
    """Represents a filter that output a ON (True) or OFF (False) boolean state only if the binary sensor has stayed in the same state for at least the specified time period.

    This filter uses two time delays: on and off, if only the first is supplied
      the delays are equal.
    """

    __slots__ = ("_off_delay", "_on_delay")

    @override
    def __init__(self, on_delay: float, off_delay: float | None) -> None:
        """Create the delayed on and off filter."""
        super().__init__()
        self._on_delay = on_delay
        self._off_delay = on_delay if off_delay is None else off_delay

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        delay = self._on_delay if value else self._off_delay
        self._set_timeout(
            self.ON_OFF, delay, lambda: self.async_output(value, is_initial)
        )
        return None

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}(on_delay={self._on_delay:0.3f}, off_delay={self._off_delay:0.3f})"


class DelayedOnFilter(Filter):
    """Represents a filter that only outputs an ON value if the binary sensor has stayed ON for at least the specified time period.

    When a ON (True) boolean state is received, wait for the specified
     time period to output it. If an OFF (False) boolean state is
     received while waiting, the original OFF station action is discarded.
    """

    __slots__ = ("_delay",)

    @override
    def __init__(self, delay: float) -> None:
        """Create the delayed on filter."""
        super().__init__()
        self._delay = delay

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        if value:
            self._set_timeout(
                self.ON, self._delay, lambda: self.async_output(True, is_initial)
            )
            return None
        self._cancel_timeout(self.ON)
        return False

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}(delay={self._delay:0.3f})"


class DelayedOffFilter(Filter):
    """Represents a filter that only outputs an OFF value if the binary sensor has stayed OFF for at least the specified time period.

    When a OFF (False) boolean state is received, wait for the specified
     time period to output it. If an ON (True) boolean state is
     received while waiting, the original OFF station action is discarded.
    """

    __slots__ = ("_delay",)

    @override
    def __init__(self, delay: float) -> None:
        """Create the delayed off filter."""
        super().__init__()
        self._delay = delay

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        if not value:
            self._set_timeout(
                self.OFF, self._delay, lambda: self.async_output(False, is_initial)
            )
            return None
        self._cancel_timeout(self.OFF)
        return True

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}(delay={self._delay:0.3f})"


class InvertFilter(Filter):
    """Represents a filter that inverts (negates) the received ON (True) or OFF (False) boolean state."""

    __slots__ = ()

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        return not value

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}()"


class LambdaFilter(Filter):
    """Represents a filter that applies a lambda function to a boolean state."""

    __slots__ = ("_function",)

    @override
    def __init__(
        self, function: Callable[[bool | None, datetime | None], bool | None]
    ) -> None:
        """Create the lambda filter with a function to apply to the boolean state."""
        super().__init__()
        self._function = function

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered by the function."""
        return self._function(value, datetime.now(UTC))

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}(function={self._function})"


class SettleFilter(Filter):
    """Represents a filter that settle an input boolean state.

    When a boolean state is received, the filter outputs it
     but wait for the received state to remain the same
     for specified time period before output any additional
     state changes.

    This filter complements the `DelayedOnOffFilter` but output value
     changes at the beginning of the delay period.
    """

    __slots__ = ("_delay", "_steady")

    @override
    def __init__(self, delay: float) -> None:
        """Create the settle filter with a delay."""
        super().__init__()
        self._delay = delay
        self._steady = True

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        if not self._steady:
            self._set_timeout(
                self.SETTLE,
                self._delay,
                lambda: self._settle_and_output(value, is_initial),
            )
            return None
        self._steady = False
        await self.async_output(value, is_initial)
        self._set_timeout(self.SETTLE, self._delay, self._restore_steady)
        return value

    def _settle_and_output(self, value, is_initial):
        async def wrapper():
            self._steady = True
            await self.async_output(value, is_initial)

        return wrapper()

    def _restore_steady(self):
        async def wrapper():
            self._steady = True

        return wrapper()

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}(delay={self._delay:0.3f})"


class AutoRepeatFilterTiming:
    """Represent the timing parameters for auto-repeat behavior in `AutoRepeatFilter`.

    Specifies the initial delay before repeating and the ON/OFF durations for subsequent
    cycles. Used as part of a sequence in `AutoRepeatFilter` to create multi-stage
    auto-repeat patterns.
    """

    __slots__ = ("_delay", "_time_off", "_time_on")

    def __init__(
        self, delay: float = 1.0, time_on: float = 0.1, time_off: float = 0.9
    ) -> None:
        """Create the timing parameters for auto-repeat behavior."""
        self._delay = delay
        self._time_on = time_on
        self._time_off = time_off

    def __repr__(self):
        """Return the timing parameters class string representation."""
        return f"{self.__class__.__name__}(delay={self._delay:0.3f}, time_on={self._time_on:0.3f}, time_off={self._time_off:0.3f})"


class AutoRepeatFilter(Filter):
    """Represent a filter with  auto-repeat behavior. The filter is parametrized by a list of timing descriptions (`AutoRepeatFilterTiming`).

    When a boolean state ON (True).

    The first timing description is taken.

    a) It no timming is active, ON (True) is ouputed.
    b) The delay is started. After the delay  expires OFF (False) is outputed.
    c) A second delay of on_delay seconds is started. When expired ON (True) is outputed.
    d) A third delay of off_delay seconds is started. When expired OFF (True) is outputed,

    This process continues, moving through the timing phases, until the last phase is reached,
    after which the last phase's timing is used indefinitely.

    If another boolean state ON (True) is received while any timer is active, it will be ignored.

    Receiving an OFF (False) input stops the whole process and immediately
    outputs OFF (False).

    Receiving an OFF (False) input stops the whole process and immediately
    outputs OFF (False).

    The filter with no timing description is equivalent to one timing with all the
    parameters set to default values (see `AutoRepeatFilterTiming` defaults).

    `AutoRepeatFilter` transforms a single ON input into a timed sequence of ON/OFF outputs,
    following a programmable pattern, and stops the sequence immediately on receiving
    an OFF input. This is useful for generating pulse or blink patterns from a
    single ON event.
    """

    __slots__ = ("_active_timing", "_timings")

    @override
    def __init__(self, timings: list[AutoRepeatFilterTiming] | None = None) -> None:
        """Create the auto-repeat filter with a list of timing descriptions."""
        super().__init__()
        timings = timings or []
        self._timings = timings
        self._active_timing = 0

    @override
    async def async_new_value(self, value: bool, is_initial: bool) -> bool | None:
        """Receive the ON (True) or OFF (False) boolean state value and returns it filtered."""
        if value:
            if self._active_timing != 0:
                return None
            await self._async_next_timing()
            return True
        self._cancel_timeout(self.TIMING)
        self._cancel_timeout(self.ON_OFF)
        self._active_timing = 0
        return False

    async def _async_next_timing(self):
        """Start the next timing in the sequence."""
        if self._active_timing < len(self._timings):
            self._set_timeout(
                self.TIMING,
                self._timings[self._active_timing]._delay,  # noqa: SLF001
                lambda: self._next_timing_wrapper(),
            )
        if self._active_timing <= len(self._timings):
            self._active_timing += 1
        if self._active_timing == 2:
            await self._async_next_value(False)

    def _next_timing_wrapper(self):
        """Wrap a call the next timing asynchronously."""
        return self._async_next_timing()

    async def _async_next_value(self, val: bool):
        """Output the next value in the auto-repeat sequence."""
        timing = self._timings[self._active_timing - 2]
        await self.async_output(val, False)
        self._set_timeout(
            self.ON_OFF,
            timing._time_on if val else timing._time_off,  # noqa: SLF001
            lambda: self._next_value_wrapper(not val),
        )

    def _next_value_wrapper(self, val: bool):
        """Wrap the call the next value asynchronously."""
        return self._async_next_value(val)

    @override
    def __repr__(self):
        """Return the filter class string representation."""
        return f"{self.__class__.__name__}(timings={self._timings!r})"


class FilteredBinarySensor(BinarySensorEntity):
    """Represent a filtered binary sensor entity."""

    hass: HomeAssistant  # Overridden BinarySensorEntity Homeassistant instance
    _event_type: BinarySensorDeviceClass  # Type of the event sensor
    _state: bool  # Current state of the sensor
    _initial_state: bool  # Initial state of the sensor
    _has_state: bool  # Whether the sensor has a state
    _state_callbacks: list[bool, Callable[[bool], None]]  # List of state callbacks
    _filter_list: Filter | None = None  # List of filters applied to the sensor
    _publish_dedup: DeduplicationHelper  # Helper for deduplicating state changes
    _async_publish_initial_state: bool  # Whether to publish the initial state
    _last_updated: datetime | None  # Last time the state update was requested
    _check_interval: bool  # Interval for checking the state
    _unsub_interval: Callable[[], None] | None  # Callback timer subscription function
    _check_function: CoroutineType[
        Any, Any, None
    ]  # Callback function called every check interval

    def __init__(
        self,
        hass: HomeAssistant,
        check_interval: float = 0.0,
        initial_state: bool = False,
        publish_initial_state: bool = True,
    ) -> None:
        """Create a filtered binary sensor entity."""
        super().__init__()
        self.hass: HomeAssistant = hass
        self._initial_state: bool = initial_state
        self._state: bool = initial_state
        self._has_state: bool = False
        self._state_callbacks: list[bool, Callable[[bool], None]] = []
        self._filter_list: Filter | None = None
        self._publish_dedup: DeduplicationHelper = DeduplicationHelper()
        self._async_publish_initial_state: bool = publish_initial_state
        self._last_updated: datetime | None = None
        self._check_interval: float = check_interval
        self._unsub_interval: Callable[[], None] | None = None
        self._check_function: CoroutineType[Any, Any, None] = self._async_check_state

    @property
    @override
    def is_on(self) -> bool:
        """Return if the sensor is ON."""
        return self._has_state and self._state

    @property
    def is_off(self) -> bool:
        """Return if the sensor is OFF."""
        return self._has_state and not self._state

    @property
    def on_initial_state(self) -> bool:
        """Return True if the current state is the initial state."""
        return self._state == self._initial_state

    def add_state_callback(self, callback_: Callable[[bool], None]):
        """Add callback to the list as tuple with is_coroutine flag and the callback function."""
        self._state_callbacks.append(
            (asyncio.iscoroutinefunction(callback_), callback_)
        )

    def clear_state_callbacks(self):
        """Clear the state callbacks list."""
        self._state_callbacks = []

    def add_filter(self, filter: Filter):
        """Add a filter to the filter list."""
        filter._parent = self  # noqa: SLF001
        if self.hass is not None:
            filter._task_creator = self.hass.create_task  # noqa: SLF001
        if self._filter_list is None:
            self._filter_list = filter
        else:
            current = self._filter_list
            while current.next is not None:  # noqa: SLF001
                current = current.next  # noqa: SLF001
            current.next = filter  # noqa: SLF001

    def add_filters(self, filters: list[Filter]):
        """Add a list of filters to the filter list."""
        for f in filters:
            self.add_filter(f)

    async def async_clear_filters(self):
        """Clear the filter list."""
        if self._filter_list is not None:
            current = self._filter_list
            await current.async_cleanup()
            while current.next is not None:  # noqa: SLF001
                current = current.next  # noqa: SLF001
                await current.async_cleanup()
            self._filter_list = None

    async def _async_publish_initial_state(self, state: bool):
        """Publish the initial state of the sensor."""
        if not self._publish_dedup.next(state):
            return
        if self._filter_list is None:
            await self._async_send_state_internal(state, is_initial=True)
        else:
            await self._filter_list.async_input(state, is_initial=True)

    async def _async_send_state_internal(self, state: bool, is_initial: bool):
        """Send the state to the callbacks."""
        self._has_state = True
        self._state = state
        if not is_initial or self._async_publish_initial_state:
            for is_coroutine, cb in self._state_callbacks:
                if is_coroutine:
                    await cb(state, is_initial)
                else:
                    cb(state, is_initial)

    async def async_reset(self):
        """Reset the sensor to its initial state."""
        self._state = self._initial_state
        self._has_state = True
        await self.async_update_ha_state(force_refresh=True)

    async def _async_publish_state(self, arg1=None, arg2=None):
        """Publish the current state of the sensor."""
        self.async_write_ha_state()

    async def async_update_state(self, state: bool = False):
        """Update the state of the sensor and publish using the configured callback."""
        self._last_updated = datetime.now(UTC)
        if not self._publish_dedup.next(state):
            return
        if self._filter_list is None:
            await self._async_send_state_internal(state, is_initial=False)
        else:
            await self._filter_list.async_input(state, is_initial=False)

    async def _async_check_state(self, now=None):
        """Check if state is ON and last update was more than check interval seconds ago."""
        if self._state and self._last_updated:
            time_since_update = (datetime.now(UTC) - self._last_updated).total_seconds()
            if time_since_update > self._check_interval:
                _LOGGER.info(
                    "State of %s is ON but last update was more than %.1f seconds ago, restoring state to initial state",
                )
                await self.async_reset()

    @override
    async def async_added_to_hass(self) -> None:
        """Run when entity is added to HA."""
        await super().async_added_to_hass()
        # Add state callback
        self.add_state_callback(self._async_publish_state)
        # Start checking state
        if self._check_interval > 0.0:
            self._unsub_interval = async_track_time_interval(
                self.hass,
                self._check_function,
                timedelta(seconds=self._check_interval),
            )
        await self.async_reset()

    @override
    async def async_will_remove_from_hass(self) -> None:
        """Clean up when entity is removed."""
        await super().async_will_remove_from_hass()
        # Remove unsub interval
        if self._unsub_interval is not None:
            self._unsub_interval()
            self._unsub_interval = None
        # Remove states callback functions
        self.clear_state_callbacks()
        # Clear filters cancelling runnings tasks
        await self.async_clear_filters()

    def __repr__(self):
        """Return the string representation of the sensor."""
        filter_list = []
        if self._filter_list is not None:
            current = self._filter_list
            filter_list.append(repr(current))
            while current.next is not None:  # noqa: SLF001
                current = current.next  # noqa: SLF001
                filter_list.append(repr(current))
        filter_list = "⇒".join(filter_list)
        return f"{self.__class__.__name__}(dedup={self._publish_dedup!r}, filters=[{filter_list}], check_interval={self._check_interval:0.3f}, publish_initial_state={self._async_publish_initial_state})"


class EventSensor(FilteredBinarySensor):
    """Represent an event sensor."""

    @override
    def __init__(
        self,
        channel: "Channel",
        event_type: BinarySensorDeviceClass,
        on_off_delay: int = 0,
        inverse: bool = False,
    ) -> None:
        """Create and event sensor entity."""
        super().__init__(
            channel.hass,
            check_interval=30.0,
            initial_state=inverse,
            publish_initial_state=True,
        )
        self._channel: Channel = channel
        self._event_type = event_type

        # Configure FilteredBinarySensor
        if on_off_delay is not None:
            self.add_filter(
                DelayedOnOffFilter(on_delay=on_off_delay, off_delay=on_off_delay)
            )
        if inverse:
            self.add_filter(InvertFilter())

        self._enabled = self._channel._enabled  # noqa: SLF001
        self._unique_id: str = f"{self._channel._unique_id}_{self._event_type}"  # noqa: SLF001
        self._key: str = f"{self._channel._key}:{self._event_type}"  # noqa: SLF001

        self._attr_device_class = event_type
        self._attr_unique_id = self._unique_id

        # Register entity
        self.hass.data[DOMAIN][self._entry_id][self._subentry_id]["entities"][
            self.unique_id
        ] = self

    @property
    def _path(self) -> tuple[int, str, int, BinarySensorDeviceClass]:
        return (
            self.device.service.port,
            self.device.serial,
            self.channel_no,
            self.event_type,
        )

    @property
    def service(self) -> HomeAssistant:
        """Return event sensor channel device service."""
        return self.channel.device.service

    @property
    def device(self) -> Device:
        """Return event sensor channel device."""
        return self.channel.device

    @property
    def channel(self) -> Device:
        """Return event sensor channel."""
        return self._channel

    @property
    def event_type(self) -> Device:
        """Return event sensor event."""
        return self._event_type

    @property
    def _entry_id(self) -> str:
        """Return config entry id."""
        return self.channel._entry_id  # noqa: SLF001

    @property
    def _subentry_id(self) -> str:
        """Return channel device subentry id."""
        return self.channel._subentry_id  # noqa: SLF001

    @property
    def serial(self) -> str:
        """Return channel device serial."""
        return self._channel.serial

    @property
    def channel_no(self) -> bool:
        """Return the state of the sensor."""
        return self.channel.channel_no

    @property
    @override
    def name(self) -> str:
        """Return the name of the sensor."""
        suffix = str(self.device_class).title()
        return f"{self.channel._name} {suffix}"  # noqa: SLF001

    @property
    @override
    def entity_registry_visible_default(self) -> bool:
        """Return if the entity should be visible when first added."""
        return self._enabled

    @property
    @override
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.channel._unique_id)},  # noqa: SLF001
            name=self.channel._name,  # noqa: SLF001
            manufacturer="DVR Alarm Service",
            model="DVR Alarm Channel",
            serial_number=f"{self.serial}-{self.channel_no}",
            suggested_area=self.channel._area,  # noqa: SLF001
            via_device=(DOMAIN, self.device._unique_id),  # noqa: SLF001
        )


class RunningSensor(FilteredBinarySensor):
    """Represent a running sensor."""

    @override
    def __init__(
        self,
        service: "Service",
    ) -> None:
        """Create a running sensor entity."""
        super().__init__(
            service.hass,
            check_interval=30.0,
            initial_state=False,
            publish_initial_state=True,
        )

        self._service: Service = service
        self.add_filter(LambdaFilter(lambda s, t: s))

        self._event_type = BinarySensorDeviceClass.RUNNING
        self._unique_id: str = f"{self.service._unique_id}_{self._event_type}"  # noqa: SLF001
        self._key: str = f"{self.service._key}_{self._event_type}"  # noqa: SLF001

        self._attr_device_class = BinarySensorDeviceClass.RUNNING
        self._attr_unique_id = self._unique_id

        # Register entity
        self.hass.data[DOMAIN][service._entry_id]["entities"][self._unique_id] = self

    @property
    def _entry_id(self) -> str:
        """Return config entry id."""
        return self.service._entry_id  # noqa: SLF001

    @property
    def service(self):
        """Return the service associated with this sensor."""
        return self._service

    @property
    @override
    def name(self) -> str:
        """Return the name of the sensor."""
        suffix = str(self.device_class).title()
        return f"{self.service._name} {suffix}"  # noqa: SLF001

    @property
    @override
    def device_info(self) -> DeviceInfo:
        """Return device info."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.service._unique_id)},  # noqa: SLF001
            name=self.service._name,  # noqa: SLF001
            manufacturer="DVR Alarm Service",
            model="DVR Alarm TCP Server",
        )

    @override
    async def _async_check_state(self, now=None):
        """Check if state is ON and last update was more than check interval seconds ago."""
        if self._state and self._last_updated:
            time_since_update = (datetime.now(UTC) - self._last_updated).total_seconds()
            if time_since_update > self._check_interval:
                if not self._service.is_running:
                    _LOGGER.warning(
                        "Service %s is not running, will try to restart it",
                        self.service._key,  # noqa: SLF001
                    )
                    await self._service.async_start()
                else:
                    await self.async_update_state(self._initial_state)


class Service:
    """Representation of a service."""

    _hass: HomeAssistant
    _entry_id: str
    _port: int
    _timeout: int
    _buffer_size: int
    _devices: dict[str, Device]
    _packet_callback: Callable[[HomeAssistant, "Service", dict[str, Any]], None] | None
    _server: asyncio.Server
    _unique_id: str
    _key = str
    _name: str
    _state_sensor: RunningSensor | None
    _state: bool

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry = None,
        packet_callback: Callable[[HomeAssistant, "Service", dict[str, Any]], None]
        | None = None,
    ) -> None:
        """Create a TCP DVR Alarm server."""
        self.hass: HomeAssistant = hass
        self._entry_id = config_entry.entry_id
        service_config = config_entry.data
        self._port = service_config.get(CONF_SERVICE_PORT, DEFAULT_SERVICE_PORT)
        self._timeout = service_config.get(
            CONF_SERVICE_TIMEOUT, DEFAULT_SERVICE_TIMEOUT
        )
        self._buffer_size = service_config.get(
            CONF_SERVICE_BUFFER_SIZE, DEFAULT_SERVICE_BUFFER_SIZE
        )
        self._name: str = (
            service_config.get(CONF_SERVICE_NAME)
            or f"DVR Alarm Service on port {self.port}"
        )
        self._devices: dict[str, Device] = {}

        self._packet_callback = packet_callback
        self._server: asyncio.Server | None = None

        self._unique_id: str = f"{DOMAIN}_{self._entry_id}_{self.port}"
        self._key = f"{self.port:05}"

        self._state_sensor: RunningSensor | None = RunningSensor(self)
        self._state = False

        # Register class
        hass.data[DOMAIN][self._entry_id][DATA_ALARM_SERVER] = self

        for config_subentry in config_entry.subentries.values():
            hass.data[DOMAIN][self._entry_id].setdefault(
                config_subentry.subentry_id, {}
            )
            hass.data[DOMAIN][self._entry_id][config_subentry.subentry_id].setdefault(
                "entities", {}
            )
            self.add_device(Device(self, config_subentry=config_subentry))

    @property
    def _path(self) -> tuple[int]:
        return (self.port,)

    @property
    def port(self) -> int:
        """Return the port number used by the service."""
        return self._port

    @property
    def timeout(self) -> int:
        """Return the timeout value for the service."""
        return self._timeout

    @property
    def buffer_size(self) -> int:
        """Return the buffer size for the service."""
        return self._buffer_size

    @property
    def is_running(self) -> bool:
        """Return if the service is running."""
        return self._state

    @property
    def devices(self) -> dict[str, "Device"]:
        """Return the devices dictionary for the service."""
        return self._devices

    @property
    def event_sensors(self) -> list["EventSensor"]:
        """Return a dictinary ofevent sensors used by this service."""
        event_sensors: list[EventSensor] = []
        for device in self.devices.values():
            device: Device
            for channel in device.channels.values():
                event_sensors.extend(channel.sensors.values())
        return event_sensors

    def add_device(self, device: Device) -> Device:
        """Add a device to the service."""
        if device.serial not in self.devices:
            self.devices[device.serial] = device
        return self.devices[device.serial]

    def get_entities(
        self, include_state=False
    ) -> dict[str, list[Union["EventSensor", "RunningSensor"]]]:
        """Return all sensors entities used by the service grouped by subentry_id."""
        entities = {}
        if include_state and self._state_sensor is not None:
            entities[None] = [self._state_sensor]
        for event_sensor in self.event_sensors:
            subentry_id = getattr(event_sensor, "_subentry_id", None)
            if subentry_id not in entities:
                entities[subentry_id] = []
            entities[subentry_id].append(event_sensor)
        return entities

    def get_device_registries(self) -> list[dict[str, Any]]:
        """Return a list of device registry keyword arguments."""
        return [device._get_device_registry() for device in self.devices.values()]  # noqa: SLF001

    def setup_device(
        self,
        subentry_id: str,
        serial: str,
        channel_count: int = 1,
        name: str | None = None,
        area: str | None = None,
        delay: int = 0,
    ) -> str:
        """Configure a device."""
        device = Device(self, subentry_id, serial, channel_count, name, area, delay)
        return self.add_device(device)

    def setup_device_channel(
        self,
        serial,
        channel: int,
        enabled: bool = True,
        name: str | None = None,
        area: str | None = None,
    ) -> str | None:
        """Create a device channel for this service."""
        if serial in self.devices:
            return self.devices[serial].setup_channel(channel, enabled, name, area)
        return None

    async def _async_handle_payload(self, payload) -> None:
        """Handle payload."""
        _LOGGER.info(
            "Service %s handle message: serial=%s, type=%s, channel=%s, event=%s, status=%s, address=%s, description=%s",
            self._key,
            payload.get("SerialID"),
            payload.get("Channel"),
            payload.get("Type"),
            payload.get("Event"),
            payload.get("Status"),
            payload.get("Address"),
            payload.get("Descrip"),
        )
        event = payload["Event"]
        event_types = set()
        for (
            payload_event,
            sensor_types,
        ) in PAYLOAD_EVENT_MAPPING.items():
            if event == payload_event:
                event_types.update(sensor_types)

        serial, channel_no = (payload[k] for k in ("SerialID", "Channel"))

        sensors = getattr(
            getattr(
                getattr(self, "devices", {}).get(serial, object), "channels", {}
            ).get(channel_no, object),
            "sensors",
            {},
        )
        if sensors:
            new_state = payload["Status"] == "Start"
            for sensor in (s for e, s in sensors.items() if e in event_types):
                sensor: EventSensor
                await sensor.async_update_state(new_state)
            status_events = {
                BinarySensorDeviceClass.CONNECTIVITY,
                BinarySensorDeviceClass.PROBLEM,
            }
            if not event_types.intersection(status_events):
                for sensor in (s for e, s in sensors.items() if e in status_events):
                    if not sensor.on_initial_state:
                        await sensor.reset()

    async def _async_handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection."""
        addr = writer.get_extra_info("peername")
        client_id = f"{addr[0]}:{addr[1]}"
        _LOGGER.info("Service %s client connection: from=%s", self._key, client_id)

        try:
            # Read header
            header_data = await reader.readexactly(PACKET_HEADER_SIZE)
            header = self._parse_header(header_data)
            if not header:
                return

            head, version, session, seq_num, msg_id, data_len = header

            # Validate header
            if not (
                head == EXPECTED_HEAD
                and version == EXPECTED_VERSION
                and session == EXPECTED_SESSION
                and seq_num == EXPECTED_SEQUENCE
                and msg_id == EXPECTED_MSG_ID
            ):
                _LOGGER.warning(
                    "Service %s invalid header from %s: head=%s, version=%s, session=%s, seq_num=%s, msg_id=%s, data_len=%s",
                    self._key,
                    client_id,
                    head,
                    version,
                    session,
                    seq_num,
                    msg_id,
                    data_len,
                )
                return

            # Read JSON data
            json_data = await reader.readexactly(data_len)
            json_data = json_data.rstrip(b"\x00")
            try:
                payload = json.loads(json_data.decode("utf-8"))
                if all(
                    k in payload for k in ["SerialID", "Channel", "Event", "Status"]
                ):
                    # NOTE The payload channel is 0-based
                    payload["Channel"] = payload["Channel"] + 1
                    if "StartTime" in payload:
                        payload["StartTime"] = self._parse_datetime(
                            payload["StartTime"]
                        )
                    else:
                        payload["StartTime"] = datetime.now()
                    if "Address" in payload:
                        payload["Address"] = self._parse_address(payload["Address"])
                    else:
                        payload["Address"] = ipaddress.ip_address(addr[0])
                    if payload["Event"] in PAYLOAD_EVENT_MAPPING:
                        await self._async_handle_payload(payload)
                    else:
                        _LOGGER.info(
                            "Service %s unhandled message: serial=%s, channel=%s, type=%s, event=%s, status=%s, address=%s, description=%s",
                            self._key,
                            payload.get("SerialID"),
                            payload.get("Channel"),
                            payload.get("Type"),
                            payload.get("Event"),
                            payload.get("Status"),
                            payload.get("Address"),
                            payload.get("Descrip"),
                        )
                else:
                    _LOGGER.warning(
                        "Service %s invalid payload: %s", self._key, payload
                    )
                if self._packet_callback:
                    self._packet_callback(self.hass, self, payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as err:
                _LOGGER.error("Service %s payload parse failed: %s", self._key, err)

        except (TimeoutError, asyncio.IncompleteReadError):
            _LOGGER.debug(
                "Service %s client connection timeout or incomplete packet: from=%s",
                self._key,
                client_id,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.error(
                "Service %s error handling client from=%s: %s",
                self._key,
                client_id,
                err,
            )
        finally:
            writer.close()
            await writer.wait_closed()

    async def async_start(self) -> bool:
        """Start servicing."""
        try:
            self._server = await asyncio.start_server(
                self._async_handle_client,
                "0.0.0.0",
                self.port,
            )
            _LOGGER.info("Service %s server started on port %d", self._key, self.port)
            self._state = True
        except Exception as err:  # noqa: BLE001
            _LOGGER.error(
                "Service %s failed to start server on port %d: %s",
                self._key,
                self.port,
                err,
            )
            self._state = False
        if self._state_sensor:
            await self._state_sensor.async_update_state(self._state)
        return self._state

    async def async_stop(self) -> bool:
        """Stop servicing."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            _LOGGER.info("Service %s server stoped on port %d", self._key, self.port)
        self._state = False
        if self._state_sensor:
            await self._state_sensor.async_update_state(self._state)
        return self._state

    @staticmethod
    def _parse_header(data: bytes) -> tuple[int, int, int, int, int, int] | None:
        """Parse the 20-byte header."""
        try:
            return struct.unpack(PACKET_HEADER_FORMAT, data)
        except struct.error as err:
            _LOGGER.error("Service packet header unpack failed: %s", err)

    @staticmethod
    def _parse_address(hex_str) -> IPv4Address | None:
        """Convert hex IP string to dotted notation."""
        try:
            hex_clean = hex_str.strip().upper().replace("0X", "")
            if not hex_clean:
                return None
            ip_int = int(hex_clean, 16)
            return str(IPv4Address(ip_int.to_bytes(4, "little")))
        except (ValueError, AttributeError) as err:
            _LOGGER.error("Service ip address  parsing failed: %s", err)
            return None

    @staticmethod
    def _parse_datetime(datetime_str: str) -> str | None:
        """Parse payload date and time."""
        formats = (
            "%Y-%m-%d %H:%M:%S",  # Space separator
            "%Y-%m-%dT%H:%M:%S",  # T separator
            "%Y-%m-%d%H:%M:%S",  # No separator
        )
        for fmt in formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        _LOGGER.error("Service datetime parsing failed: '%s'", datetime_str)
        return None
