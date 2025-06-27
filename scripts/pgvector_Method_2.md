# Guide: Installing pgvector on Windows (The Simple Pre-compiled Method)

This guide provides the simplest, most reliable method for installing `pgvector` on Windows by using pre-compiled files. This bypasses all complex build steps involving Visual Studio and `NMAKE`.

**This guide replaces all previous installation instructions.**

---

### Step 1: Clean Up Any Previous Attempts

To prevent any conflicts, we must remove any files left over from the old build process.

1.  **Delete the source folder:**
    *   Open File Explorer, type `%TEMP%` in the address bar, and press Enter.
    *   Find and **delete the `pgvector` folder**.

2.  **Delete old installed files:**
    *   Navigate to your PostgreSQL installation's `lib` folder (e.g., `C:\Program Files\PostgreSQL\17\lib`).
    *   Delete `vector.dll` if it exists.
    *   Navigate to the `share\extension` folder (e.g., `C:\Program Files\PostgreSQL\17\share\extension`).
    *   Delete `vector.control` and all `vector--*.sql` files if they exist.

---

### Step 2: Download the Correct Pre-compiled ZIP File

1.  Go to the official `pgvector` releases page on GitHub:
    [**https://github.com/pgvector/pgvector/releases**](https://github.com/pgvector/pgvector/releases)

2.  Scroll down to the "Assets" section of the latest release.

3.  Find the file that matches your PostgreSQL version. Since you are using PostgreSQL 17, you need the file named `pgvector-vX.Y.Z-pg17-win64.zip`.
    *   **Example:** For version 0.8.0, the file is `pgvector-v0.8.0-pg16-win64.zip`. Find the one for `pg17`. *Note: As of this writing, `pg17` might not have an official binary. If you cannot find one, you may need to downgrade PostgreSQL to version 16, which is fully supported.*

4.  Download this `.zip` file to your computer.

---

### Step 3: Copy Files to PostgreSQL Folders

1.  **Unzip the file** you just downloaded. You will see two folders inside: `lib` and `share`.

2.  **Copy the contents into your PostgreSQL installation directory:**
    *   Open the unzipped `lib` folder. Copy the `pgvector.dll` file.
    *   Navigate to your PostgreSQL installation's `lib` folder (e.g., `C:\Program Files\PostgreSQL\17\lib`) and **paste the file there**.
    *   Go back to the unzipped `share` folder and open its `extension` subfolder. Copy **all files** inside it.
    *   Navigate to your PostgreSQL installation's `share\extension` folder (e.g., `C:\Program Files\PostgreSQL\17\share\extension`) and **paste the files there**.

---

### Step 4: Restart the PostgreSQL Service (CRITICAL STEP)

The server must be restarted to recognize the new files.

1.  Open the Windows Start Menu.
2.  Type `services.msc` and press Enter. This will open the Services management console.
3.  Scroll down the list to find your PostgreSQL service (e.g., `postgresql-x64-17`).
4.  **Right-click** on the service and select **"Restart"**.

---

### Step 5: Run the Master Launcher Again

Now, the setup script will finally succeed.

1.  Go back to your project's terminal.
2.  Make sure your virtual environment is active (`.\venv\Scripts\activate`).
3.  Run the master launcher script:
    ```bash
    python run_prometheus.py
    ```

This time, the restarted PostgreSQL server will recognize the pre-compiled `pgvector` files, and the setup check will pass.