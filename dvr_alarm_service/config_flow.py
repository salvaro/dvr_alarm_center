"""Config flow for the dvr_alarm_service integration."""

from __future__ import annotations

from ipaddress import ip_address
import logging
import re
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector

from .const import (
    CONF_CHANNEL_AREA,
    CONF_CHANNEL_DEVICE_ID,
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
    CONF_SERVICE_PORT,
    CONF_SERVICE_TIMEOUT,
    CONF_TITLE,
    DEFAULT_SERVICE_BUFFER_SIZE,
    DEFAULT_SERVICE_PORT,
    DEFAULT_SERVICE_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__package__)

STEP_SERVICE_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SERVICE_PORT, default=DEFAULT_SERVICE_PORT): vol.All(
            vol.Coerce(int), vol.Range(min=1500, max=65535)
        ),
        vol.Required(CONF_SERVICE_TIMEOUT, default=DEFAULT_SERVICE_TIMEOUT): vol.All(
            vol.Coerce(int), vol.Range(min=15, max=120)
        ),
        vol.Required(
            CONF_SERVICE_BUFFER_SIZE, default=DEFAULT_SERVICE_BUFFER_SIZE
        ): vol.All(vol.Coerce(int), vol.Range(min=1024, max=8192)),
    }
)


def STEP_DEVICE_DATA_SCHEMA(current: dict[str, Any] | None = None):
    """Device data input schema."""
    current = current or {}
    current = {k: v for k, v in current.items() if v is not None}
    return vol.Schema(
        {
            vol.Required(
                CONF_DEVICE_ID, default=current.get(CONF_DEVICE_ID, vol.UNDEFINED)
            ): str,
            vol.Required(
                CONF_DEVICE_CHANNEL_COUNT,
                default=current.get(CONF_DEVICE_CHANNEL_COUNT, 1),
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=64)),
            vol.Required(
                CONF_DEVICE_SENSOR_DELAY,
                default=current.get(CONF_DEVICE_SENSOR_DELAY, 0),
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=600)),
            vol.Optional(
                CONF_DEVICE_NAME, default=current.get(CONF_DEVICE_NAME, vol.UNDEFINED)
            ): str,
            vol.Optional(
                CONF_DEVICE_AREA, default=current.get(CONF_DEVICE_AREA, vol.UNDEFINED)
            ): selector.AreaSelector(),
        }
    )


def STEP_DEVICE_CHANNEL_DATA_SCHEMA(
    channel_count: int | None = 1, current: dict[str, Any] | None = None
):
    """Device channels data input schema."""
    channel_count = channel_count or 1
    current = current or {}
    current = current.get(CONF_DEVICE_CHANNELS, {})
    for channel_no in current:
        current[channel_no] = {
            k: v for k, v in current[channel_no].items() if v is not None
        }
    return vol.Schema(
        {
            vol.Optional(f"channel_{channel_no}_options"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            f"channel_{channel_no}_enabled",
                            default=current.get(channel_no, {}).get(
                                CONF_CHANNEL_ENABLED, True
                            ),
                        ): bool,
                        vol.Optional(
                            f"channel_{channel_no}_name",
                            default=current.get(channel_no, {}).get(
                                CONF_CHANNEL_NAME, vol.UNDEFINED
                            ),
                        ): str,
                        vol.Optional(
                            f"channel_{channel_no}_area",
                            default=current.get(channel_no, {}).get(
                                CONF_CHANNEL_AREA, vol.UNDEFINED
                            ),
                        ): selector.AreaSelector(),
                    }
                ),
                {"collapsed": True},
            )
            for channel_no in range(1, channel_count + 1)
        }
    )


def _is_valid_device_serial(device_id: str) -> bool:
    """Validate device ID format (IP or serial number)."""
    try:
        # Check for IP address format
        ip_address(device_id)
    except ValueError:
        # Check for serial format
        serial_regex = r"[a-zA-Z0-9_-]{8,32}"
        return re.match(serial_regex, device_id) is not None


async def update_service(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate service input."""
    data = data or {}
    service_data = {
        CONF_SERVICE_PORT: data.get(CONF_SERVICE_PORT, DEFAULT_SERVICE_PORT),
        CONF_SERVICE_TIMEOUT: data.get(CONF_SERVICE_TIMEOUT, DEFAULT_SERVICE_TIMEOUT),
        CONF_SERVICE_BUFFER_SIZE: data.get(
            CONF_SERVICE_BUFFER_SIZE, DEFAULT_SERVICE_BUFFER_SIZE
        ),
    }
    service_data[CONF_TITLE] = (
        f"DVR Alarm Service Server on port {service_data[CONF_SERVICE_PORT]}"
    )
    return service_data


async def update_device(
    hass: HomeAssistant, user_data: dict[str, Any], data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate device input."""
    data = data or {}
    serial = user_data.get(CONF_DEVICE_ID)
    if not _is_valid_device_serial(serial):
        raise InvalidDeviceSerial
    channel_count = user_data.get(CONF_DEVICE_CHANNEL_COUNT, 1)
    if channel_count < 1 or channel_count > 64:
        raise InvalidDeviceChannelCount
    channels_data = user_data.get(CONF_DEVICE_CHANNELS, {})
    for channel_no in range(1, channel_count + 1):
        channel_data = user_data.get(CONF_DEVICE_CHANNELS, {}).get(channel_no, {})
        channels_data[channel_no] = {
            CONF_CHANNEL_DEVICE_ID: serial,
            CONF_CHANNEL_ID: channel_no,
            CONF_CHANNEL_ENABLED: channel_data.get(CONF_CHANNEL_ENABLED, True),
            CONF_CHANNEL_NAME: channel_data.get(CONF_CHANNEL_NAME),
            CONF_CHANNEL_AREA: channel_data.get(CONF_CHANNEL_AREA),
        }
    channels_data.update(data.get(CONF_DEVICE_CHANNELS, {}))
    return {
        CONF_DEVICE_ID: serial,
        CONF_DEVICE_CHANNEL_COUNT: channel_count,
        CONF_DEVICE_SENSOR_DELAY: user_data.get(CONF_DEVICE_SENSOR_DELAY, 0),
        CONF_DEVICE_NAME: user_data.get(CONF_DEVICE_NAME),
        CONF_DEVICE_AREA: user_data.get(CONF_DEVICE_AREA),
        CONF_DEVICE_CHANNELS: channels_data,
        CONF_TITLE: f"Device {user_data[CONF_DEVICE_NAME]}"
        if user_data.get(CONF_DEVICE_NAME)
        else f"Device {serial}",
    }


async def update_device_channels(
    hass: HomeAssistant, user_data: dict[str, Any], data: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Validate device channels input."""
    serial = user_data.get(CONF_DEVICE_ID)
    device_title = (
        f"Device {user_data[CONF_DEVICE_NAME]}"
        if user_data.get(CONF_DEVICE_NAME)
        else f"Device {serial}"
    )
    if not _is_valid_device_serial(serial):
        raise InvalidDeviceSerial
    channel_count = user_data.get(CONF_DEVICE_CHANNEL_COUNT, 1)
    if channel_count < 1 or channel_count > 64:
        raise InvalidDeviceChannelCount
    # device_area = device_data.get(CONF_DEVICE_AREA)
    # device_name = device_data.get(CONF_DEVICE_NAME, device_id)
    channel_data = {}
    for channel_no in range(1, channel_count + 1):
        prefix = f"channel_{channel_no}_"
        section_data = {
            k.removeprefix(prefix): v
            for k, v in data.get(f"{prefix}options", {}).items()
        }
        if section_data:
            channel_data[channel_no] = {
                CONF_CHANNEL_DEVICE_ID: serial,
                CONF_CHANNEL_ID: channel_no,
                CONF_CHANNEL_ENABLED: section_data.get(CONF_CHANNEL_ENABLED, True),
                CONF_CHANNEL_NAME: section_data.get(CONF_CHANNEL_NAME),
                CONF_CHANNEL_AREA: section_data.get(CONF_CHANNEL_AREA),
                CONF_TITLE: section_data[CONF_CHANNEL_NAME]
                if section_data.get(CONF_CHANNEL_NAME)
                else f"{device_title} Channel #{channel_no}",
            }
    user_data[CONF_DEVICE_CHANNELS].update(channel_data)
    return user_data


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for dvr_alarm_service."""

    VERSION = 1

    def __init__(self) -> None:
        """Create device_config property."""
        super().__init__()
        self.service_config: dict[str, Any] = {}
        self.device_config: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        # if await self._async_check_service_configured():
        #     return await self.async_step_device()
        return await self.async_step_service()

        # errors: dict[str, str] = {}
        # if user_input is not None:
        #     try:
        #         info = await validate_input(self.hass, user_input)
        #     except CannotConnect:
        #         errors["base"] = "cannot_connect"
        #     except InvalidAuth:
        #         errors["base"] = "invalid_auth"
        #     except Exception:
        #         _LOGGER.exception("Unexpected exception")
        #         errors["base"] = "unknown"
        #     else:
        #         return self.async_create_entry(title=info["title"], data=user_input)

        # return self.async_show_form(
        #     step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        # )

    async def async_step_service(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                self.service_config = await update_service(self.hass, user_input)
                await self._async_check_duplicate_service_port()
            except DuplicateServicePort:
                errors["base"] = "duplicate_service_port"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title=self.service_config[CONF_TITLE],
                    data=self.service_config,
                    # unique_id=f"{DOMAIN}_{self.service_config[CONF_SERVICE_PORT]}",
                )
                # return await self.async_step_device()

        return self.async_show_form(
            step_id="service", data_schema=STEP_SERVICE_DATA_SCHEMA, errors=errors
        )

    async def _async_check_duplicate_service_port(self) -> bool:
        """Check if service port is already taken."""
        unique_id = self.service_config[CONF_SERVICE_PORT]
        if any(
            entry.data.get(CONF_SERVICE_PORT) == unique_id
            for entry in self.hass.config_entries.async_entries(self.handler)
        ):
            raise DuplicateServicePort
        return True

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {"device": DeviceSubentryFlow}


class DeviceSubentryFlow(ConfigSubentryFlow):
    """Handle subentry flow for adding and modifying a device and its channels."""

    def __init__(self) -> None:
        """Create device_config property."""
        super().__init__()
        self.service_config: dict[str, Any] = {}
        self.device_config: dict[str, Any] = {}
        self.unique_id: str | None = None

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None):
        """Add reconfigure step to allow to reconfigure a config entry."""
        reconfigure_subentry = self._get_reconfigure_subentry()
        if reconfigure_subentry:
            self.device_config = dict(reconfigure_subentry.data)
            self.device_config[CONF_DEVICE_CHANNELS] = {
                int(k): v for k, v in self.device_config[CONF_DEVICE_CHANNELS].items()
            }
            self.unique_id = reconfigure_subentry.unique_id
        return await self.async_step_device()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the initial step."""
        return await self.async_step_device()
        # await self.hass.config_entries.async_reload(self.config_entry.entry_id)

    async def async_step_device(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                self.device_config = await update_device(
                    self.hass, user_input, self.device_config
                )
                if self.source != SOURCE_RECONFIGURE:
                    await self._async_check_duplicate_device_id()
            except DuplicateDeviceSerial:
                errors["base"] = "duplicate_device_id"
            except InvalidDeviceSerial:
                errors["base"] = "invalid_device_id"
            except InvalidDeviceChannelCount:
                errors["base"] = "invalid_device_channels"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return await self.async_step_device_channels()

        return self.async_show_form(
            step_id="device",
            data_schema=STEP_DEVICE_DATA_SCHEMA(self.device_config),
            errors=errors,
            last_step=False,
        )

    async def async_step_device_channels(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                self.device_config = await update_device_channels(
                    self.hass, self.device_config, user_input
                )
            except InvalidDeviceSerial:
                errors["base"] = "invalid_device_id"
            except InvalidDeviceChannelCount:
                errors["base"] = "invalid_device_channels"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                if self.source != SOURCE_RECONFIGURE:
                    self.hass.config_entries.async_schedule_reload(self._entry_id)
                    return self.async_create_entry(
                        title=self.device_config[CONF_TITLE],
                        data=self.device_config,
                        unique_id=self.device_config[CONF_DEVICE_ID],
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    unique_id=self.unique_id,
                    title=self.device_config[CONF_TITLE],
                    data=self.device_config,
                )

        data_schema = STEP_DEVICE_CHANNEL_DATA_SCHEMA(
            self.device_config[CONF_DEVICE_CHANNEL_COUNT], self.device_config
        )
        return self.async_show_form(
            step_id="device_channels",
            data_schema=data_schema,
            # description_placeholders=description_placeholders,
            errors=errors,
            last_step=True,
        )

    async def _async_check_duplicate_device_id(self) -> bool:
        """Check if device is already configured."""
        # Get parent config entry
        unique_id = self.device_config[CONF_DEVICE_ID]
        if any(
            entry.unique_id == unique_id
            for entry in self.hass.config_entries.async_entries(self.handler)
        ):
            raise DuplicateDeviceSerial
        return True


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""


class InvalidDeviceSerial(HomeAssistantError):
    """Error to indicate there is invalid device id (serial number)."""


class DuplicateServicePort(HomeAssistantError):
    """Error to indicate this is a service port is already taken."""


class DuplicateDeviceSerial(HomeAssistantError):
    """Error to indicate this is a duplicate device serial."""


class InvalidDeviceChannelCount(HomeAssistantError):
    """Error to indicate there is invalid device channel count."""
