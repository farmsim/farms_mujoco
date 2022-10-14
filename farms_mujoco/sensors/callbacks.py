""" Sensor callbacks for mujoco """

import numpy as np
from dm_control.mujoco.wrapper import mjbindings
from farms_core.sensors.sensor_convention import sc
from farms_muscle import rigid_tendon as rt


def mjcb_sensor(mj_model, mj_data, stage):
    """ mujoco sensor callback """

    # mujoco enums
    mj_enums = mjbindings.enums

    # number of user sensors
    nu_sensors = mj_model.nuser_sensor
    n_sensors = mj_model.nsensor
    # user sensor ids
    user_sensors_id = np.nonzero(
            mj_model.sensor_type == mj_enums.mjtSensor.mjSENS_USER
    )[0]
    #
    # from IPython import embed; embed()
    # muscle_sensors_id = mj_model.user_sensor_id[np.nonzero(
    #      mj_model.sensor_user[:] == 0
    # )[1]]
    for sensor_id in user_sensors_id:
        # get sensor info
        sensor_objid = mj_model.sensor_objid[sensor_id]
        sensor_ftype = mj_model.sensor_user[sensor_id][0]
        sensor_adr = mj_model.sensor_adr[sensor_id]
        stim = mj_data.ctrl[sensor_objid]
        l_mtu = mj_data.actuator_length[sensor_objid]
        v_mtu = mj_data.actuator_velocity[sensor_objid]
        muscle_force = mj_data.actuator_force[sensor_objid]
        gainprm = mj_model.actuator_gainprm[sensor_objid]
        f_max = gainprm[0]
        l_opt = gainprm[1]
        l_slack = gainprm[2]
        v_max = gainprm[3]
        alpha_opt = gainprm[4]
        alpha = rt.c_pennation_angle(l_mtu, l_opt, l_slack, alpha_opt)

        if sensor_ftype == sc.muscle_pennation_angle:
            mj_data.sensordata[sensor_adr] = alpha
        elif sensor_ftype == sc.muscle_fiber_length:
            mj_data.sensordata[sensor_adr] = rt.c_fiber_length(
                l_mtu, l_slack, alpha
            )/l_opt
        # elif sensor_ftype == sc.muscle_fiber_velocity:
        #     mj_data.sensordata[sensor_adr] = rt.c_fiber_velocity(
        #         v_mtu, alpha
        #     )/v_max
        # elif sensor_ftype == sc.muscle_Ib_feedback:
        #     Ib_kF = mj_model.actuator_user[sensor_objid][8]
        #     muscle_Ib = Ib_kF*muscle_force/f_max
        #     mj_data.sensordata[sensor_adr] = muscle_Ib
        # elif sensor_ftype == sc.muscle_Ia_feedback:
        #     Ia_kv = mj_model.actuator_user[sensor_objid][0]
        #     Ia_pv = mj_model.actuator_user[sensor_objid][1]
        #     Ia_k_dI = mj_model.actuator_user[sensor_objid][2]
        #     Ia_k_nI = mj_model.actuator_user[sensor_objid][3]
        #     Ia_const_I = mj_model.actuator_user[sensor_objid][4]
        #     l_ce = rt.c_fiber_length(l_mtu, l_slack, alpha)/l_opt
        #     v_ce = rt.c_fiber_velocity(v_mtu, alpha)/v_max
        #     Ia_aff = Ia_kv*abs(v_ce)**Ia_pv + Ia_k_dI*l_ce + Ia_k_nI*stim + Ia_const_I
            #     mj_data.sensordata[sensor_adr] = Ia_aff
