import os
import mysql.connector
import urllib.parse

def get_db_connection():
    mysql_url = os.environ.get('MYSQL_URL')
    if mysql_url:
        parsed = urllib.parse.urlparse(mysql_url)
        config = {
            'host': parsed.hostname,
            'user': parsed.username,
            'password': parsed.password,
            'database': parsed.path[1:],
            'port': parsed.port or 3306
        }
        return mysql.connector.connect(**config)
    else:
        # Configuración local
        config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'walmart_db')
        }
        return mysql.connector.connect(**config)