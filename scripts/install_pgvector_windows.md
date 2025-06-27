# Guide: Installing pgvector on Windows for PostgreSQL (Manual Copy Method)

This guide provides a more direct, manual method for installing `pgvector` to resolve the `extension "pgvector" is not available` error. This bypasses potential issues with the `nmake install` script.

## Prerequisites

- **Visual Studio:** Must be installed with the **"Desktop development with C++"** workload.
- **Administrator Privileges:** You will need to run a command prompt as an administrator.

---

### Step 1: Clean Up Previous Attempts

To ensure a clean slate, we must remove any old files from previous attempts.

1.  **Delete the source folder:**
    *   Open File Explorer and navigate to your temp directory by typing `%TEMP%` in the address bar and pressing Enter.
    *   Find and **delete the `pgvector` folder**.

2.  **Delete old installed files (if they exist):**
    *   Navigate to your PostgreSQL installation's `lib` folder (e.g., `C:\Program Files\PostgreSQL\17\lib`).
    *   Delete `vector.dll` if it exists.
    *   Navigate to the `share\extension` folder (e.g., `C:\Program Files\PostgreSQL\17\share\extension`).
    *   Delete `vector.control` and all `vector--*.sql` files if they exist.

---

### Step 2: Build the `vector.dll` File

1.  **Open the Correct Command Prompt:**
    *   Open the Windows Start Menu, type `x64 Native Tools`, and **right-click "x64 Native Tools Command Prompt for VS..."** to **"Run as administrator"**.

2.  **Set PostgreSQL Root Path:**
    *   **Change the version number (`17` in the example) to match your own.**
    ```cmd
    set "PGROOT=C:\Program Files\PostgreSQL\17"
    ```

3.  **Download and Build:**
    *   Navigate to your temp directory, clone the repository, and build the DLL.
    ```cmd
    cd %TEMP%
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
    cd pgvector
    nmake /F Makefile.win
    ```
    *   After this completes, a file named `vector.dll` will be created inside the `C:\Users\<YourUser>\AppData\Local\Temp\pgvector` directory.

---

### Step 3: Manually Copy the Extension Files

This is the most critical part. We will copy the built files to the correct PostgreSQL directories.

1.  **Open two File Explorer windows:**
    *   **Window 1 (Source):** Navigate to the build directory, `C:\Users\<YourUser>\AppData\Local\Temp\pgvector`.
    *   **Window 2 (Destination):** Navigate to your PostgreSQL installation directory, `C:\Program Files\PostgreSQL\<Your_Version>`.

2.  **Copy `vector.dll`:**
    *   In Window 1 (source), find `vector.dll`.
    *   Copy it.
    *   In Window 2 (destination), go into the `lib` folder.
    *   Paste `vector.dll` here.

3.  **Copy `vector.control`:**
    *   In Window 1 (source), find `vector.control`.
    *   Copy it.
    *   In Window 2 (destination), go into the `share\extension` folder.
    *   Paste `vector.control` here.

4.  **Copy the SQL Files:**
    *   In Window 1 (source), go into the `sql` folder.
    *   Select **all** `.sql` files inside this folder.
    *   Copy them.
    *   In Window 2 (destination), go into the `share\extension` folder.
    *   Paste all the `.sql` files here.

At the end of this step, your `C:\Program Files\PostgreSQL\17\share\extension` folder should contain `vector.control` and many `vector--...sql` files. Your `C:\Program Files\PostgreSQL\17\lib` folder should contain `vector.dll`.

---

### Step 4: Restart the PostgreSQL Service

The server must be restarted to recognize the new files.

1.  Open the Windows Start Menu.
2.  Type `services.msc` and press Enter.
3.  Scroll down to find your PostgreSQL service (e.g., `postgresql-x64-17`).
4.  **Right-click** on the service and select **"Restart"**.

---

### Step 5: Run the Master Launcher Again

Now, the setup script should finally succeed.

1.  Go back to your project's terminal.
2.  Make sure your virtual environment is active (`.\venv\Scripts\activate`).
3.  Run the master launcher script:
    ```bash
    python run_prometheus.py
    ```

This time, the check should pass, and the application will proceed.