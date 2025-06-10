# Home Assistant Custom Component: DVR-IP/Sofia Alarm Center Sensors  

<!---
[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg)](https://github.com/hacs/integration)
-->

**Monitor motion and occupancy using alarm messages from low-cost DVR-IP/Sofia surveillance systems.**  

This custom component listens for alarm center messages from **Sofia-powered DVRs, NVRs, and IP cameras** (with alarm center capability) and creates corresponding motion/occupancy sensors in Home Assistant for each device channel.  

## ⚙️ Prerequisites  
Before setup, ensure your surveillance devices are properly configured:  
1. **Enable Alarm Center** using (*requires admin credentials*):  
   - The manufacturer's CMS application  
   - Tools like [`sofiactl`](https://gitlab.com/667bdrm/sofiactl)  
2. **Configure Alarm Server Settings** on each device:  
   - **Server Address**: Your Home Assistant instance’s local IPv4 address  
   - **Port**: Must match the port you’ll set in this component  
3. **Note Device Details**:  
   - Record the **serial number** and **channel number** (IP Cameras are typically single-channel)  

## ⚠️ Security Notice  
- Alarm messages are **unencrypted**.  
- **Restrict service ports** to your local network (block external access).  
- Malicious devices on your LAN could send spoofed alerts—use in trusted environments only.  

## 📥 Installation  
1. Copy the `custom_components/dvr_alarm_service` folder to your Home Assistant config directory.  
2. Restart Home Assistant.  






