""" Run Gazebo island """

import os
import subprocess
import sqlite3
# import time


def run_simulation(
        world_path="/.gazebo/models/salamander_new/world.world",
        database='salamander.db'
):
    """ Run island """
    port = 11345
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    # Create table
    with conn:
        cursor.execute(
            "CREATE TABLE"
            " IF NOT EXISTS gazebo_address"
            " (address text, UNIQUE(address))"
        )
        conn.commit()
    # Insert address into table

    print("Verifying addresses")
    with conn:
        offset = 0
        addresses_used = [
            address_used[0]
            for address_used
            in cursor.execute('SELECT * FROM gazebo_address')
        ]
        print("Verifying addresses_used: {}".format(addresses_used))
        for _ in range(1000):
            address = "localhost:{}".format(port+offset)
            if address in addresses_used:
                offset += 1
            else:
                print("Found unused address {}".format(address))
                break
        cursor.execute("INSERT INTO gazebo_address VALUES (?)", (address,))
        conn.commit()
    print("Verification of addresses complete")

    print("Addresses before simulation:")
    for _address in cursor.execute('SELECT * FROM gazebo_address'):
        print("    {}".format(_address))

    exe = "gzserver"
    verbose = "--verbose"
    os.environ["GAZEBO_MASTER_URI"] = address
    cmd = "{} {} {}".format(
        exe,
        verbose,
        os.environ["HOME"]+world_path
    )
    print(cmd)
    subprocess.call(cmd, shell=True)
    print("Simulation complete")

    # Delete address from database
    with conn:
        cursor.execute("DELETE FROM gazebo_address WHERE address=(?)", (address,))
        conn.commit()

    print("Addresses after simulation:")
    for _address in cursor.execute('SELECT * FROM gazebo_address'):
        print("    {}".format(_address))

    # # We can also close the connection if we are done with it.
    # # Just be sure any changes have been committed or they will be lost.
    # conn.close()
