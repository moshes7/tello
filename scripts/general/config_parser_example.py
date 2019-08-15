import configparser

config = configparser.ConfigParser()
config.read('config.ini')

print(config['general']['speed'])
print(config['control_reference']['x'])
print(config['pid']['kp_x'])

print('Done')