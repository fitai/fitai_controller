# mqtt_py

*Server controls*

This repo is located in `/var/opt/python/mqtt_py`. Not sure if we want to keep it there, or in `/opt/python...` 

## Handling Dependencies 

To run the script, the proper virtualenvironment needs to be activated. Unfortunately, for now, the virtualenvs are all on my user (jbrubaker).. to get them installed on your user you may need to install anaconda.

*pip* can be used to install all requirements from *requirements.txt*

e.g.
`cd /var/opt/python/mqtt; pip install -r requirements.txt`

If any packages fail, I use `conda` (through anaconda) as a backup. 

`conda install <package_name>`

## Code Flow 

As of right now, the script is simple. It listens to the MQTT broker on localhost port 1883. When a message is pushed to the broker, the code is notified and receives a payload, e.g. the data packet. This payload is in a specific JSON format, which the code knows to convert to a python dict. Utility functions parse the dict into two separate pandas dataframes - the header, which contains all relevant metadata and the content, which contains all the accelerometer values.

## Running the script 

As of now, there isn't much to do here

`cd /var/opt/python/mqtt_py; python main.py`

The script will print a few messages to console as it receives packets. If any errors occur, they should print to console and be caught by the try/except statements. If this fails, the script will break and have to be restarted.
