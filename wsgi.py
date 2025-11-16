import sys
path = '/home/<nikhilsimha>/house_price_api'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
