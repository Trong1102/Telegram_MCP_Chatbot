import os
import json
import pymysql
import sshtunnel
from dotenv import load_dotenv

# --- 1. Load environment variables ---
load_dotenv()

SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 22))
SSH_USER = os.getenv("SSH_USER")
SSH_KEY_FILE = os.getenv("SSH_KEY_FILE")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

LOCAL_PORT = int(os.getenv("LOCAL_PORT", 33006))

# --- 2. Connect via SSH Tunnel ---
def get_db_schema():
    with sshtunnel.SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USER,
        ssh_pkey=SSH_KEY_FILE,
        remote_bind_address=(MYSQL_HOST, MYSQL_PORT),
        local_bind_address=('127.0.0.1', LOCAL_PORT)
    ) as tunnel:
        print(f"🔐 SSH Tunnel established at localhost:{LOCAL_PORT}")

        # --- 3. Connect MySQL ---
        connection = pymysql.connect(
            host='127.0.0.1',
            port=LOCAL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor
        )
        print("✅ Connected to MySQL!")

        schema = []
        try:
            with connection.cursor() as cursor:
                # --- 4. Get all tables ---
                cursor.execute("SHOW TABLES;")
                tables = [row[f'Tables_in_{MYSQL_DATABASE}'] for row in cursor.fetchall()]

                for table in tables:
                    cursor.execute(f"SHOW COLUMNS FROM `{table}`;")
                    columns = cursor.fetchall()
                    schema.append({
                        "table_name": table,
                        "columns": columns
                    })
            
            # --- 5. Save schema to file ---
            with open("schema.json", "w", encoding="utf-8") as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
            
            print("💾 Schema saved to schema.json")
            
            # Print simple summary
            for t in schema:
                print(f"\n📦 Table: {t['table_name']}")
                for c in t['columns']:
                    print(f"  - {c['Field']} ({c['Type']})")

        finally:
            connection.close()
            print("🔌 MySQL connection closed.")

if __name__ == "__main__":
    get_db_schema()
