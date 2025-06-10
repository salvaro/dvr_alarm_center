"""The dvr_alarm_service integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntry

from .binary_sensor import Service
from .const import DATA_ALARM_SERVER, DATA_CONFIG, DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__package__)


_PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR]


type ServerConfigEntry = ConfigEntry[Service]


async def async_setup(hass: HomeAssistant, config: ConfigEntry) -> bool:
    """Set up the component."""
    hass.data.setdefault(DOMAIN, {})
    # server = DVRAlarmServer(hass, _handle_packet)
    # hass.data.setdefault(DOMAIN, {DATA_ALARM_SERVER: server})

    # from homeassistant.helpers import device_registry as dr

    # device_registry = dr.async_get(hass)

    # device = device_registry.async_get_or_create(
    #     config_entry_id=f"{DOMAIN}_{DATA_ALARM_SERVER}",
    #     identifiers={(DOMAIN, f"{DOMAIN}_{DATA_ALARM_SERVER}")},
    #     manufacturer="DVR Alarm Service",
    #     name="DVR Alarm Service Server",
    # )
    # await server.start()

    # server = DVRAlarmServer(hass, _handle_packet)
    # hass.data.setdefault(DOMAIN, {})
    # hass.data.setdefault(DOMAIN, {DATA_ALARM_SERVER: server})

    # async def _async_setup_entities(
    #     hass: HomeAssistant, async_add_entities: AddEntitiesCallback
    # ) -> None:
    #     """Add entities safely.Se in the event loop
    # hass.async_add_job(
    #     _async_setup_entities,
    #     hass,
    #     lambda entities: hass.async_create_task(
    #         hass.data[DOMAIN][Platform.BINARY_SENSOR].async_add_entities(entities)
    #     ),
    # )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up dvr_alarm_service from a config entry."""

    hass.data[DOMAIN][entry.entry_id] = {
        DATA_CONFIG: entry.data,
    }

    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)

    return True


async def async_remove_config_entry_device(
    hass: HomeAssistant, config_entry: ConfigEntry, device_entry: DeviceEntry
) -> bool:
    """Remove a config entry from a device."""
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    service: Service | None = hass.data[DOMAIN][entry.entry_id].get(DATA_ALARM_SERVER)
    if service is not None:
        await service.async_stop()
        service._state_sensor = None  # noqa: SLF001
        for device in service.devices.values():
            for channel in device.channels.values():
                for _sensor in channel.sensors.values():
                    pass
                channel._event_sensors = {}  # noqa: SLF001
            device._channels = {}  # noqa: SLF001
        service._devices = {}  # noqa: SLF001

    unload_ok = await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok
