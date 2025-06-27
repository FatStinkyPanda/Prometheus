# Prometheus - PostgreSQL Setup Guide

This guide will walk you through the necessary steps to correctly configure the PostgreSQL database user and password for the Prometheus application. The most common startup error is `password authentication failed`, which this guide will resolve.

## The Goal

The objective is to ensure the password for the `prometheus_app` user in your PostgreSQL database **exactly matches** the password specified in the `backend/config/prometheus_config.yaml` file.

---

### Step 1: Set the Password in PostgreSQL

You need to connect to your database with administrative privileges to change the password for the `prometheus_app` user.

#### For Windows Users:

1.  Open the **"SQL Shell (psql)"** application, which was installed with PostgreSQL.
2.  You will be prompted for Server, Database, Port, and Username. Press **Enter** for each one to accept the defaults (e.g., Server: `localhost`, Database: `postgres`, Port: `5432`, Username: `postgres`).
3.  Enter the password for your `postgres` superuser when prompted. This is the password you created when you installed PostgreSQL.
4.  You should now be at a `postgres=#` prompt.

#### For macOS / Linux Users:

1.  Open your terminal.
2.  Connect to PostgreSQL using the `postgres` superuser account. The command is typically:
    ```bash
    sudo -u postgres psql
    ```
3.  You should now be at a `postgres=#` prompt.

#### All Users - The SQL Command:

Once you are at the `postgres=#` prompt, type the following command and press Enter.

**IMPORTANT:** Replace `'YourNewSecurePassword'` with a strong, memorable password of your choice.

```sql
ALTER USER prometheus_app WITH PASSWORD 'Ft1zL964X76!';