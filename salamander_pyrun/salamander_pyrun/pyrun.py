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
    print("Preparing {} simulation".format(world_path))
    port = 11345
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    # Create table
    with conn:
        cursor.execute(
            "CREATE TABLE"
            " IF NOT EXISTS gazebo_address"
            " (address text, model text, UNIQUE(address))"
        )
        conn.commit()
    # Insert address into table

    print("Verifying addresses")
    address_found = False
    with conn:
        addresses_used = [
            address_used[0]
            for address_used
            in cursor.execute('SELECT * FROM gazebo_address')
        ]
        print("Verifying addresses_used: {}".format(addresses_used))
        for offset in range(1000):
            address = "localhost:{}".format(port+offset)
            if address not in addresses_used:
                print("Found unused address {}".format(address))
                try:
                    cursor.execute(
                        "INSERT INTO gazebo_address VALUES (?, ?)",
                        (address, world_path)
                    )
                    conn.commit()
                except sqlite3.IntegrityError as err:
                    print(err)
                    print("Insertion into {} failed, trying another address")
                else:
                    address_found = True
            if address_found:
                break
            if offset == 999:
                raise Exception("No available address found")
    print("Verification of addresses complete")

    print("Addresses before simulation:")
    for _address in cursor.execute('SELECT * FROM gazebo_address'):
        print("    {}".format(_address))

    exe = "gzserver"
    verbose = "--verbose"
    seed = "--seed 0"
    minimal_comms = "" # "--minimal_comms"
    os.environ["GAZEBO_MASTER_URI"] = address
    cmd = "{} {} {} {} {}".format(
        exe,
        verbose,
        seed,
        minimal_comms,
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
