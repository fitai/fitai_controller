#!/bin/sh

# website startup
/usr/bin/nodejs /var/www/html/laravel/socket.js > /opt/socket_js.log &

# pubsub broker daemon
/usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf &

# python data packet handler
/home/kyle/virtualenvs/fitai/bin/python /opt/fitai_controller/comms/mqtt_client.py -v > /opt/mqtt_client.log &

# python RFID packet handler
/home/kyle/virtualenvs/fitai/bin/python /opt/fitai_controller/comms/rfid_client.py -v > /opt/rfid_client.log &
