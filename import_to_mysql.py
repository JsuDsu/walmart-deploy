import mysql.connector
import urllib.parse

# URL pública de Railway
MYSQL_PUBLIC_URL = "mysql://root:XDysOVebJCMyWBziOttUjTeaEQUXbosL@turntable.proxy.rlwy.net:25238/railway"

# Archivo SQL exportado
SQL_FILE = "walmart_db.sql"

def import_sql():
    # Parsear URL de Railway
    parsed = urllib.parse.urlparse(MYSQL_PUBLIC_URL)

    config = {
        'host': parsed.hostname,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path[1:],
        'port': parsed.port or 3306
    }

    try:
        # Conexión
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        print("✅ Conectado a Railway MySQL")

        # Leer archivo SQL
        with open(SQL_FILE, "r", encoding="utf-8") as file:
            sql_script = file.read()

        # Separar consultas
        queries = sql_script.split(';')

        # Ejecutar una por una
        for query in queries:
            query = query.strip()

            if query:
                try:
                    cursor.execute(query)
                except Exception as e:
                    print(f"⚠️ Error en consulta:\n{query[:100]}...")
                    print(e)

        conn.commit()

        print("✅ Base de datos importada correctamente.")

    except mysql.connector.Error as err:
        print(f"❌ Error de conexión: {err}")

    finally:
        if 'cursor' in locals():
            cursor.close()

        if 'conn' in locals():
            conn.close()

        print("🔌 Conexión cerrada.")

if __name__ == "__main__":
    import_sql()