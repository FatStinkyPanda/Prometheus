# run_prometheus.py
# This is the single, master entry point for the entire Prometheus application.
# It checks for a valid environment and orchestrates all necessary setup steps.
#
# Password Management:
# - PostgreSQL master password is saved to .pg_master_password (add to .gitignore!)
# - Application user password is automatically synchronized with config file
# - Use --reset-password to clear saved password and prompt again

import sys
import os
import venv
import subprocess
import getpass
import asyncio
import stat
from pathlib import Path

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

def get_venv_executable(venv_path: Path, executable_name: str) -> str:
    """Gets the path to an executable inside the venv."""
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / executable_name)
    else:
        return str(venv_path / "bin" / executable_name)

def get_password_file_path(project_root: Path) -> Path:
    """Get the path to the PostgreSQL master password file."""
    return project_root / ".pg_master_password"

def read_master_password(project_root: Path) -> str:
    """Read the PostgreSQL master password from file."""
    password_file = get_password_file_path(project_root)
    try:
        with open(password_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        print_color(f"Warning: Could not read password file: {e}", Colors.WARNING)
        return None

def save_master_password(project_root: Path, password: str):
    """Save the PostgreSQL master password to file with secure permissions."""
    password_file = get_password_file_path(project_root)
    try:
        # Create the file with restrictive permissions
        with open(password_file, 'w', encoding='utf-8') as f:
            f.write(password)
        
        # Set secure file permissions (readable only by owner)
        if sys.platform != "win32":
            os.chmod(password_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
        else:
            # On Windows, try to restrict access (best effort)
            try:
                import win32security
                import win32api
                # Get current user
                user_sid = win32security.LookupAccountName(None, win32api.GetUserName())[0]
                
                # Create a new security descriptor
                sd = win32security.SECURITY_DESCRIPTOR()
                # Create a DACL
                dacl = win32security.ACL()
                # Grant full control to the current user
                dacl.AddAccessAllowedAce(win32security.ACL_REVISION, 0x001f01ff, user_sid) # GENERIC_ALL
                # Set the DACL to the security descriptor
                sd.SetSecurityDescriptorDacl(1, dacl, 0)
                # Apply the security descriptor to the file
                win32security.SetFileSecurity(str(password_file), win32security.DACL_SECURITY_INFORMATION, sd)
            except ImportError:
                print_color("Warning: win32security module not found. Windows file permissions not applied to password file.", Colors.WARNING)
            except Exception as e_win_perm:
                print_color(f"Warning: Could not set secure permissions on password file (Windows): {e_win_perm}", Colors.WARNING)
        
        print_color(f"Master password saved securely to: {password_file}", Colors.OKGREEN)
        print_color("Note: This file contains your PostgreSQL superuser password.", Colors.OKCYAN)
        print_color("Keep it secure and do not commit it to version control.", Colors.OKCYAN)
        
        # Update .gitignore to exclude password file
        _ensure_gitignore_excludes_password_file(project_root)
        
    except Exception as e:
        print_color(f"Warning: Could not save password file: {e}", Colors.WARNING)

def _ensure_gitignore_excludes_password_file(project_root: Path):
    """Ensure .gitignore excludes the password file."""
    gitignore_path = project_root / '.gitignore'
    password_entry = '.pg_master_password'
    
    try:
        # Read existing .gitignore
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if password file is already excluded
            if password_entry in content:
                return  # Already present
            
            # Add the entry
            with open(gitignore_path, 'a', encoding='utf-8') as f:
                f.write(f'\n# PostgreSQL master password file\n{password_entry}\n')
        else:
            # Create new .gitignore
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(f'# PostgreSQL master password file\n{password_entry}\n')
        
        print_color("Updated .gitignore to exclude password file.", Colors.OKBLUE)
        
    except Exception as e:
        print_color(f"Warning: Could not update .gitignore: {e}", Colors.WARNING)

def get_master_password(project_root: Path) -> str:
    """Get the PostgreSQL master password, either from file or by prompting user."""
    # Try to read from file first
    password = read_master_password(project_root)
    if password:
        print_color("Using saved PostgreSQL master password.", Colors.OKBLUE)
        return password
    
    # If no file exists, prompt user and save
    print_color("No saved PostgreSQL master password found.", Colors.WARNING)
    print("You'll need to enter the PostgreSQL superuser password once.")
    print("This will be saved securely for future use.")
    
    superuser = os.getenv("PROMETHEUS_DATABASE_SUPERUSER", "postgres")
    print_color(f"Please enter the password for PostgreSQL user '{superuser}':", Colors.OKCYAN)
    password = getpass.getpass()
    
    # Save the password for future use
    save_master_password(project_root, password)
    
    return password

async def sync_app_user_password(conn, config_password: str):
    """Synchronize the prometheus_app user password with the config file."""
    try:
        print_color("Synchronizing application user password with configuration...", Colors.OKBLUE)
        
        # --- FIX: Construct SQL string with password directly ---
        # Ensure the password in the YAML doesn't contain single quotes, or escape them properly if it could.
        # For passwords from config, we assume they are safe or that user ensures they are.
        # A better way for arbitrary passwords would be to use a function that quotes them for SQL.
        # However, 'ALTER USER ... PASSWORD' expects a literal string.
        # Basic single quote escaping:
        escaped_config_password = config_password.replace("'", "''")
        await conn.execute(f"ALTER USER prometheus_app WITH PASSWORD '{escaped_config_password}';")
        
        print_color("Application user password synchronized successfully.", Colors.OKGREEN)
        
    except Exception as e:
        print_color(f"Warning: Could not update application user password: {e}", Colors.WARNING)
        print("You may need to manually update the password in your config file or database.")


async def test_master_password(project_root: Path, password: str, db_host: str, db_port: str, superuser: str) -> bool:
    """Test if the master password is correct."""
    try:
        import asyncpg
        conn = await asyncpg.connect(
            user=superuser, 
            password=password, 
            host=db_host, 
            port=db_port, 
            database='postgres' # Connect to a default DB that should always exist
        )
        await conn.close()
        return True
    except (asyncpg.exceptions.InvalidPasswordError, asyncpg.exceptions.ClientCannotConnectError, ConnectionRefusedError, OSError):
        return False
    except Exception as e:
        print_color(f"Unexpected error testing master password: {e}", Colors.WARNING)
        return False


def is_in_venv():
    """Checks if the script is running inside a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def setup_python_environment(project_root: Path):
    """Creates a venv and installs all Python dependencies using the setup_backend.py script."""
    print_color("\n--- Setting up Python Environment using setup_backend.py ---", Colors.HEADER)
    
    # Path to the setup_backend.py script
    setup_script_path = project_root / "scripts" / "setup_backend.py"
    if not setup_script_path.exists():
        print_color(f"FATAL ERROR: setup_backend.py not found at {setup_script_path}", Colors.FAIL)
        sys.exit(1)
        
    # Use the system Python to run the setup script. The setup script itself
    # will create the venv and install dependencies into it.
    # We pass --force if run_prometheus.py was called with --force-setup.
    setup_command = [sys.executable, str(setup_script_path)]
    if "--force-setup" in sys.argv:
        setup_command.append("--force")
        
    result = subprocess.run(setup_command) # No capture, let it print directly
    
    if result.returncode != 0:
        print_color("[ERROR] Python environment setup script (setup_backend.py) failed.", Colors.FAIL)
        sys.exit(1)
        
    print_color("\n[IMPORTANT] Python environment setup via setup_backend.py is complete.", Colors.BOLD + Colors.HEADER)
    venv_path = project_root / "venv"
    if sys.platform == "win32":
        activation_cmd = rf".\{venv_path.name}\Scripts\activate"
    else:
        activation_cmd = f"source {venv_path.name}/bin/activate"
    
    print("To continue, you must now activate the new environment if you are not already in it.")
    print_color(f"1. Activate it by running this command in your terminal:\n   {activation_cmd}", Colors.OKCYAN)
    print_color(f"2. Then, run the launcher again:\n   python {Path(__file__).name}", Colors.OKCYAN)
    sys.exit(0)


async def check_and_setup_database(project_root: Path):
    """Checks for a valid database and vector extension, and runs setup if needed."""
    print_color("\n--- Checking Database and Vector Extension Status ---", Colors.HEADER)
    try:
        import asyncpg
        # ConfigLoader will be imported from the venv after setup
        from backend.utils.config_loader import ConfigLoader
    except ImportError:
        # This case should not be hit if setup_python_environment ran correctly
        print_color("Critical dependency 'asyncpg' or 'ConfigLoader' is missing.", Colors.FAIL)
        print_color("This might happen if you haven't activated the venv after initial setup.", Colors.WARNING)
        sys.exit(1)

    db_host = os.getenv("PROMETHEUS_DATABASE_HOST", "localhost")
    db_port = os.getenv("PROMETHEUS_DATABASE_PORT", "5432")
    superuser = os.getenv("PROMETHEUS_DATABASE_SUPERUSER", "postgres")
    
    # Load configuration to get the app user password
    try:
        config = ConfigLoader.load_config() # Uses default primary and merge configs
        app_user_password = config.get('database', {}).get('password', 'your_secure_password_here')
    except Exception as e:
        print_color(f"Warning: Could not load config file to get app user password: {e}", Colors.WARNING)
        app_user_password = 'your_secure_password_here'
    
    print(f"This script will connect to PostgreSQL as superuser ('{superuser}') to ensure database and extensions are set up correctly.")
    
    password = get_master_password(project_root)
    
    if not await test_master_password(project_root, password, db_host, db_port, superuser):
        print_color("\nError: The saved/entered password for the PostgreSQL superuser is incorrect.", Colors.FAIL)
        password_file = get_password_file_path(project_root)
        try:
            password_file.unlink()
            print_color("Invalid password file removed.", Colors.WARNING)
        except FileNotFoundError:
            pass
        print_color("Please run the script again to enter the correct password.", Colors.WARNING)
        sys.exit(1)

    conn = None
    try:
        conn = await asyncpg.connect(user=superuser, password=password, host=db_host, port=db_port, database='postgres')
        
        db_name = config.get('database', {}).get('name', 'prometheus_db')
        db_exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if not db_exists:
            print_color(f"Database '{db_name}' not found. Creating...", Colors.OKBLUE)
            await conn.execute(f'CREATE DATABASE {db_name}')
            print_color(f"Database '{db_name}' created.", Colors.OKGREEN)
        await conn.close()
        
        conn = await asyncpg.connect(user=superuser, password=password, host=db_host, port=db_port, database=db_name)
        
        print_color("\n--- Detailed Vector Extension Diagnostic Information ---", Colors.OKBLUE)
        pg_version_info = await conn.fetchval("SHOW server_version;")
        print_color(f"PostgreSQL Version: {pg_version_info}", Colors.OKBLUE)
        
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"vector\";")
            print_color("SUCCESS: 'vector' extension is enabled (or already was).", Colors.OKGREEN)
            await conn.fetchval("SELECT '[1,2,3]'::vector;")
            print_color("SUCCESS: vector extension is functioning correctly.", Colors.OKGREEN)
        except asyncpg.exceptions.PostgresError as e_vec:
            print_color(f"\n[CRITICAL ERROR WITH VECTOR EXTENSION]", Colors.BOLD + Colors.FAIL)
            print_color(f"Full Error: {str(e_vec)} (Code: {e_vec.sqlstate})", Colors.FAIL)
            if hasattr(e_vec, 'detail') and e_vec.detail: print_color(f"Detail: {e_vec.detail}", Colors.FAIL)
            guide_path_windows = project_root / 'scripts' / 'install_pgvector_windows.md'
            guide_path_linux = project_root / 'scripts' / 'install_pgvector_linux_macos.md' # Assuming you'll create this
            guide_to_show = guide_path_windows if sys.platform == "win32" else guide_path_linux
            print_color(f"Please ensure the 'vector' extension is correctly installed for your PostgreSQL version. See: {guide_to_show}", Colors.WARNING)
            sys.exit(1)
        
        print_color("\n--- Applying Database Schema ---", Colors.OKBLUE)
        schema_file = project_root / 'backend' / 'database' / 'migrations' / 'initial_schema.sql'
        if not schema_file.exists():
            print_color(f"Schema file not found: {schema_file}", Colors.FAIL); sys.exit(1)
        with open(schema_file, 'r', encoding='utf-8') as f: schema_sql = f.read()
        await conn.execute(schema_sql)
        print_color("Database schema and 'prometheus_app' user are up to date.", Colors.OKGREEN)
        
        await sync_app_user_password(conn, app_user_password)
        
        print_color("\nDatabase setup check complete.", Colors.OKGREEN)

    except asyncpg.exceptions.InvalidPasswordError:
        print_color("\nFATAL ERROR: Incorrect password for the PostgreSQL superuser.", Colors.FAIL)
        password_file = get_password_file_path(project_root)
        try: password_file.unlink(); print_color("Invalid password file removed. Run again.", Colors.WARNING)
        except FileNotFoundError: pass
        sys.exit(1)
    except (ConnectionRefusedError, OSError) as e:
        print_color(f"\nFATAL ERROR: Connection refused. Is PostgreSQL running on {db_host}:{db_port}? ({e})", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_color(f"\nAn unexpected database error occurred: {e}", Colors.FAIL); sys.exit(1)
    finally:
        if conn and not conn.is_closed(): await conn.close()


def main():
    """The main logic for launching the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prometheus Consciousness System Launcher")
    parser.add_argument("--reset-password", action="store_true", help="Reset the saved PostgreSQL master password")
    parser.add_argument("--force-setup", action="store_true", help="Force re-run of Python environment setup script (setup_backend.py).")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root) # Ensure working directory is project root
    
    if args.reset_password:
        password_file = get_password_file_path(project_root)
        try: password_file.unlink(); print_color("Saved PostgreSQL master password has been reset.", Colors.OKGREEN)
        except FileNotFoundError: print_color("No saved password file found.", Colors.WARNING)
        sys.exit(0)

    if not is_in_venv() or args.force_setup:
        setup_python_environment(project_root) # This will exit after setup
        return

    # If in venv, make sure critical packages are importable (simple check)
    try:
        import PyQt6
        import asyncpg
        import backend.utils.config_loader 
    except ImportError as e_imp:
        print_color(f"ERROR: In virtual environment, but critical dependency '{e_imp.name}' is missing.", Colors.FAIL)
        print_color("The environment may be corrupted or incomplete. Re-running setup...", Colors.WARNING)
        setup_python_environment(project_root) # This will exit after setup
        return

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(check_and_setup_database(project_root))
    
    # --- REMOVED: asyncio.run(initialize_database_pool()) ---
    # DatabaseManager initialization is now handled within GUI or headless main paths.

    print_color("\n--- All checks passed. Starting Prometheus Application ---", Colors.HEADER)
    
    # Late import to ensure setup runs first and venv is active.
    from PyQt6.QtWidgets import QApplication
    # import qasync # qasync is imported within backend.main -> run_gui
    from backend.gui.launcher_window import LauncherWindow
    # backend.main handles both GUI and headless modes
    from backend.main import main as run_application_main 

    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    
    chosen_mode = ""
    def handle_launch_choice(option: str):
        nonlocal chosen_mode
        chosen_mode = option

    launcher.launch_option_selected.connect(handle_launch_choice)
    result = launcher.exec() # Use exec() for modal dialogs

    if result == launcher.DialogCode.Accepted and chosen_mode == "backend":
        # Prepare arguments for backend.main.main
        # GUI mode is default if --headless or --api-only are not passed.
        main_args = argparse.Namespace(headless=False, api_only=False)
        
        # qasync loop setup for GUI mode is handled inside backend/main.py -> run_gui()
        run_application_main(main_args) # This will call run_gui which sets up its own loop
    else:
        print("Launch cancelled by user or an unhandled mode was selected.")
        sys.exit(0)

if __name__ == "__main__":
    main()