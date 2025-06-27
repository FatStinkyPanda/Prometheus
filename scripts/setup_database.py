# scripts/setup_database.py

import asyncio
import getpass
import sys
from pathlib import Path

import asyncpg

# --- Helper for Colored Console Output ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(text, color):
    """Prints text in a given color."""
    print(f"{color}{text}{Colors.ENDC}")

async def main():
    """Main function to set up the PostgreSQL database."""
    print_color("--- Prometheus PostgreSQL Database Setup ---", Colors.BOLD + Colors.HEADER)
    print("This script will set up the 'prometheus_db' database and the 'prometheus_app' user.")

    # --- Step 1: Gather Connection Details ---
    db_host = input(f"Enter PostgreSQL host [default: localhost]: ") or "localhost"
    db_port = input(f"Enter PostgreSQL port [default: 5432]: ") or "5432"
    superuser = input(f"Enter PostgreSQL superuser name [default: postgres]: ") or "postgres"
    
    print_color(f"Enter password for PostgreSQL superuser '{superuser}':", Colors.OKCYAN)
    password = getpass.getpass()

    conn = None
    try:
        # --- Step 2: Connect to the default 'postgres' database to check/create our target DB ---
        print_color("\nConnecting to PostgreSQL server as superuser...", Colors.OKBLUE)
        conn = await asyncpg.connect(user=superuser, password=password, host=db_host, port=db_port, database='postgres')
        print_color("Successfully connected to the PostgreSQL server.", Colors.OKGREEN)

        # --- Step 3: Check if 'prometheus_db' database exists ---
        db_name = 'prometheus_db'
        db_exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        
        if db_exists:
            print_color(f"Database '{db_name}' already exists. Skipping creation.", Colors.OKGREEN)
        else:
            print_color(f"Database '{db_name}' not found. Creating...", Colors.OKBLUE)
            await conn.execute(f'CREATE DATABASE {db_name}')
            print_color(f"Database '{db_name}' created successfully.", Colors.OKGREEN)
        
        await conn.close() # Close connection to 'postgres' db

        # --- Step 4: Connect to 'prometheus_db' and run the schema script ---
        print_color(f"\nConnecting to '{db_name}' database to set up schema...", Colors.OKBLUE)
        conn = await asyncpg.connect(user=superuser, password=password, host=db_host, port=db_port, database=db_name)
        
        # Find and read the initial_schema.sql file
        project_root = Path(__file__).resolve().parent.parent
        schema_file = project_root / 'backend' / 'database' / 'migrations' / 'initial_schema.sql'
        
        if not schema_file.exists():
            print_color(f"FATAL ERROR: Schema file not found at '{schema_file}'", Colors.FAIL)
            sys.exit(1)
            
        print_color(f"Executing schema from: {schema_file}", Colors.OKBLUE)
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
            
        # Execute the entire script as a single transaction
        async with conn.transaction():
            await conn.execute(schema_sql)
        
        print_color("Schema, roles, and permissions have been applied successfully.", Colors.OKGREEN)
        
        # --- Final Step: Inform user about the password ---
        print_color("\n--- IMPORTANT NEXT STEP ---", Colors.BOLD + Colors.HEADER)
        print("The database user 'prometheus_app' has been created with a default password.")
        print_color("You MUST now update your configuration file to use this password.", Colors.WARNING)
        print("\n1. Open the file: " + str(project_root / 'backend' / 'config' / 'prometheus_config.yaml'))
        print("2. Find the 'database' section and change the 'password' field to:")
        print_color("   your_secure_password_here", Colors.OKCYAN + Colors.BOLD)
        print("\n(Note: For production, it's highly recommended you change this default password using the 'database_setup_guide.md'.)")

    except asyncpg.exceptions.InvalidPasswordError:
        print_color("\nFATAL ERROR: Incorrect password for the PostgreSQL superuser.", Colors.FAIL)
    except ConnectionRefusedError:
        print_color(f"\nFATAL ERROR: Connection refused. Is PostgreSQL running on {db_host}:{db_port}?", Colors.FAIL)
    except Exception as e:
        print_color(f"\nAn unexpected error occurred: {e}", Colors.FAIL)
        print("Please check your PostgreSQL server status and connection details.")
    finally:
        if conn and not conn.is_closed():
            await conn.close()
            
    print_color("\nDatabase setup script finished.", Colors.HEADER)

if __name__ == '__main__':
    # On Windows, the default asyncio event loop policy can cause issues.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())