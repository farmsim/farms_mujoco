#ifndef _SALAMANDER_PLUGIN_HH_
#define _SALAMANDER_PLUGIN_HH_

#include <memory>
#include <unordered_map>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <math.h>

#include "parameters.hh"
#include <yaml-cpp/yaml.h>


class JointOscillatorOptions : public std::unordered_map<std::string, double>
{
public:
    JointOscillatorOptions(std::string type="position")
        {
            this->type = type;
            this->insert({"amplitude", 0});
            this->insert({"frequency", 0});
            this->insert({"phase", 0});
            this->insert({"bias", 0});
            return;
        };
    virtual ~JointOscillatorOptions()
        {
            return;
        };

// Oscillator options
public:
    std::string type;
};


class Joint
{
public:
    Joint() : pid_position(
        1e1,  // _P
        1e0,  // _I
        0,  // _D
        0.1,  // _imax
        -0.1,  // _imin
        10.0,  // _cmdMax
        -10.0),  // _cmdMin
    pid_velocity(
        1e-2,  // _P
        1e-3,  // _I
        0,  // _D
        0.1,  // _imax
        -0.1,  // _imin
        10.0,  // _cmdMax
        -10.0)  // _cmdMin
        {
            return;
        }
    ~Joint()
        {
            return;
        };

public:
    std::string name;
    gazebo::physics::JointPtr joint;
    gazebo::common::PID pid_position;
    gazebo::common::PID pid_velocity;
    JointOscillatorOptions oscillator;

private:
    double position = 0;
    double velocity = 0;
    // gazebo::physics::LinkPtr parent_link_;
    // gazebo::physics::LinkPtr child_link_;

public:
    bool load(gazebo::physics::ModelPtr model)
        {
            this->joint = model->GetJoint(this->name);
            if (!this->joint)
            {
                std::cerr << "Joint not found" << std::endl;
                return false;
            }
            // this->parent_link_ = this->joint->GetParent();
            // this->child_link_ = this->joint->GetChild();
            return true;
        }

    void update()
        {
            // this->name = this->joint->GetName();
#if GAZEBO_MAJOR_VERSION >= 8
            this->position = this->joint->Position(0);
#else
            this->position = this->joint->GetAngle(0).Radian();
#endif
            this->velocity = this->joint->GetVelocity(0);
            return;
        }

    void set_force(double force)
        {
            if (!this->joint->HasType(gazebo::physics::Joint::FIXED_JOINT))
                this->joint->SetForce(0, force);
            return;
        }

    bool fixed()
        {
            return this->joint->HasType(gazebo::physics::Joint::FIXED_JOINT);
        }

    void control(double position_cmd, double velocity_cmd, bool verbose=false)
        {
            this->update();
            this->pid_position.SetCmd(0);
            double err_pos = this->position - position_cmd;
            double cmd_pos = this->pid_position.Update(
                err_pos,
                0.001);
            double err_vel = this->velocity - velocity_cmd;
            double cmd_vel = this->pid_velocity.Update(
                err_vel,
                0.001);

            if (verbose)
            {
                std::cout
                    << "Error for joint "
                    << this->name
                    << ": "
                    << err_pos
                    << std::endl;
                std::cout
                    << "Cmd for joint "
                    << this->name
                    << ": "
                    << cmd_pos
                    << " (pos)"
                    << cmd_vel
                    << " (vel)"
                    << std::endl;
            }
            this->set_force(cmd_pos+cmd_vel);
            return;
        }
};


class FT_Sensor
{
public: FT_Sensor() {}
public: ~FT_Sensor() {}

public: std::string name;
private: gazebo::physics::JointPtr joint;
private: double force[4];  // x, y, z, mag
private: double torque[4];  // x, y, z, mag
// private: gazebo::physics::LinkPtr parent_link_;
// private: gazebo::physics::LinkPtr child_link_;

public: bool load(gazebo::physics::ModelPtr model)
        {
            this->joint = model->GetJoint(this->name);
            if (!this->joint)
            {
                std::cerr << "Joint not found" << std::endl;
                return false;
            }
            // this->parent_link_ = this->joint->GetParent();
            // this->child_link_ = this->joint->GetChild();
            return true;
        }

public: gazebo::physics::JointWrench wrench()
        {
            gazebo::physics::JointWrench wrench = this->joint->GetForceTorque(0);
            ignition::math::Vector3d torque;
            ignition::math::Vector3d force;
#if GAZEBO_MAJOR_VERSION >= 8
            force = wrench.body2Force;
            torque = wrench.body2Torque;
#else
            force = wrench.body2Force.Ign();
            torque = wrench.body2Torque.Ign();
#endif
            this->force[0] = force.X();
            this->force[1] = force.Y();
            this->force[2] = force.Z();
            this->force[3] = sqrt(
                pow(force.X(), 2) + pow(force.Y(), 2) + pow(force.Z(), 2));
            this->torque[0] = torque.X();
            this->torque[1] = torque.Y();
            this->torque[2] = torque.Z();
            this->torque[3] = sqrt(
                pow(torque.X(), 2) + pow(torque.Y(), 2) + pow(torque.Z(), 2));
            return wrench;
        }
};


namespace gazebo
{

    /// \brief A plugin to control a Salamander sensor.
    class SalamanderPlugin : public ModelPlugin
    {
        /// \brief Constructor
    public: SalamanderPlugin() {}
    public: ~SalamanderPlugin()
            {
                // Deletion message
                std::cout
                    << "Model "
                    << this->model->GetName()
                    << " deleted"
                    << std::endl;
                // this->joints.clear();
                return;
            }

    private:
        // Parameters
        std::string config_filename;
        PluginParameters parameters;

    private:
        // Model information
        physics::ModelPtr model;
        std::vector<std::string> joints_names;
        std::unordered_map<std::string, std::shared_ptr<Joint>> joints;
        std::vector<std::string> ft_sensors_names;
        std::vector<FT_Sensor> ft_sensors;
        physics::WorldPtr world_;
        std::vector<std::string> frame_name_;

    private:
        // Additional information
        long step = 0;
        bool verbose = false;
        common::Time prevUpdateTime;

    private:
        // Pointer to the update event connection
        event::ConnectionPtr updateConnection;

        /// \brief The load function is called by Gazebo when the plugin is
        /// inserted into simulation
        /// \param[in] _model A pointer to the model that this plugin is
        /// attached to.
        /// \param[in] _sdf A pointer to the plugin's SDF element.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
            {
                // Store the pointer to the model
                this->model = _model;

                // Load confirmation message
                std::cout
                    << "\nThe salamander plugin is attached to model["
                    << this->model->GetName()
                    << "]"
                    << std::endl;

                this->parameters = get_parameters(_sdf);

                // Joints
                int joints_n = this->model->GetJointCount();
                std::vector<physics::JointPtr> joints_all = this->model->GetJoints();
                joints_names.resize(joints_n);
                int i = 0;
                if (this->verbose)
                    std::cout << "List of joints:" << std::endl;
                std::shared_ptr<Joint> joint_ptr;
                YAML::Node joints_parameters = this->parameters["joints"];
                for (auto &joint: joints_all) {
                    joints_names[i] = joint->GetName();
                    if (this->verbose)
                        std::cout
                            << "  "
                            << joints_names[i]
                            << " (type: "
                            << joint->GetType()
                            << ")"
                            << std::endl;
                    this->joints.insert({joints_names[i], std::make_shared<Joint>()});
                    this->joints[joints_names[i]]->name = joints_names[i];
                    if (!this->joints[joints_names[i]]->load(this->model)) {
                        return;
                    }
                    double value;
                    std::string value_str;
                    std::string name;
                    YAML::Node joint_parameters = joints_parameters[joints_names[i]];
                    if (joint->HasType(gazebo::physics::Joint::FIXED_JOINT))
                        this->ft_sensors_names.push_back(joints_names[i]);
                    else
                    {
                        // Revolute joint

                        // PID position
                        YAML::Node pid_position = joint_parameters["pid"]["position"];
                        this->joints[joints_names[i]]->pid_position.SetPGain(pid_position["p"].as<double>());
                        this->joints[joints_names[i]]->pid_position.SetIGain(pid_position["i"].as<double>());
                        this->joints[joints_names[i]]->pid_position.SetDGain(pid_position["d"].as<double>());
                        // PID velocity
                        YAML::Node pid_velocity = joint_parameters["pid"]["velocity"];
                        this->joints[joints_names[i]]->pid_velocity.SetPGain(pid_velocity["p"].as<double>());
                        this->joints[joints_names[i]]->pid_velocity.SetIGain(pid_velocity["i"].as<double>());
                        this->joints[joints_names[i]]->pid_velocity.SetDGain(pid_velocity["d"].as<double>());
                        // Oscillator
                        this->joints[joints_names[i]]->oscillator.type = joint_parameters["type"].as<std::string>();
                        this->joints[joints_names[i]]->oscillator["amplitude"] = joint_parameters["oscillator"]["amplitude"].as<double>();
                        this->joints[joints_names[i]]->oscillator["frequency"] = joint_parameters["oscillator"]["frequency"].as<double>();
                        this->joints[joints_names[i]]->oscillator["phase"] = joint_parameters["oscillator"]["phase"].as<double>();
                        this->joints[joints_names[i]]->oscillator["bias"] = joint_parameters["oscillator"]["bias"].as<double>();
                    }
                    i++;
                }
                if (this->verbose)
                    std::cout << std::endl;

                // FT sensors
                this->ft_sensors.resize(this->ft_sensors_names.size());
                i = 0;
                for (auto &name: this->ft_sensors_names) {
                    if (this->verbose)
                        std::cout << "Found FT sensor " << name << std::endl;
                    this->ft_sensors[i].name = name;
                    i++;
                }

                // Save pointers
                this->world_ = this->model->GetWorld();

                // Listen to the update event. This event is broadcast every
                // simulation iteration.
                this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&SalamanderPlugin::OnUpdate, this));
            }

        // Called by the world update start event
    public: virtual void OnUpdate()
            {
                // Time
#if GAZEBO_MAJOR_VERSION >= 8
                common::Time cur_time = this->world_->SimTime();
#else
                common::Time cur_time = this->world_->GetSimTime();
#endif
                common::Time stepTime = cur_time - this->prevUpdateTime;
                this->prevUpdateTime = cur_time;

                if(this->verbose)
                {
                    std::cout
                        << "The salamander plugin was called at step "
                        << step
                        << " and at time "
                        << cur_time
                        << " and has "
                        << this->model->GetSensorCount()
                        << " sensors"
                        << std::endl;
                    std::cout
                        << "\nThe salamander plugin is being updated "
                        << std::endl;
                }
                step++;

                int i = 0, leg_i, side_i, part_i;
                double t = cur_time.Double();
                int n_joints = 11;
                std::string name;
                std::vector<std::string> LR = {"L", "R"};

                double amplitude;
                double frequency;
                double omega;
                double phase;
                double bias;
                double pos_cmd;
                double vel_cmd;

                // Joints
                for (auto &joint: this->joints) {
                    if (!joint.second->fixed())
                    {
                        amplitude = joint.second->oscillator["amplitude"];
                        frequency = joint.second->oscillator["frequency"];
                        omega = 2*M_PI*frequency;
                        phase = joint.second->oscillator["phase"];
                        bias = joint.second->oscillator["bias"];
                        pos_cmd = amplitude*sin(omega*t - phase) + bias;
                        if (joint.second->oscillator.type == "position")
                        {
                            vel_cmd = omega*amplitude*cos(omega*t - phase);
                            joint.second->control(pos_cmd, vel_cmd);
                        }
                        else if (joint.second->oscillator.type == "torque")
                        {
                            joint.second->set_force(pos_cmd);
                        }
                        if (this->verbose) {
                            if (joint.first == "body_link_1"){
                                std::cout
                                    << "controlled "
                                    << joint.first
                                    << "(pos_cmd: "
                                    << pos_cmd
                                    << ", vel_cmd: "
                                    << vel_cmd
                                    << ")"
                                    << std::endl;
                                std::cout << "  Amplitude: " << amplitude << std::endl;
                                std::cout << "  Frequency: " << frequency << std::endl;
                                std::cout << "  Omega: " << omega << std::endl;
                                std::cout << "  Phase: " << phase << std::endl;
                                std::cout << "  Bias: " << bias << std::endl;
                            }
                        }
                    }
                }
            }
    };

// Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(SalamanderPlugin);

}

#endif
