#ifndef _SALAMANDER_PLUGIN_HH_
#define _SALAMANDER_PLUGIN_HH_

#include <memory>
#include <unordered_map>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <math.h>


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


class ExperimentLogger
{
public:
    ExperimentLogger();
    virtual ~ExperimentLogger();

    std::vector<gazebo::physics::JointWrench> wrench;
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

        // Model information
    private: physics::ModelPtr model;
    private: std::vector<std::string> joints_names;
    private: std::unordered_map<std::string, std::shared_ptr<Joint>> joints;
    private: std::vector<std::string> ft_sensors_names;
    private: std::vector<FT_Sensor> ft_sensors;
    private: physics::WorldPtr world_;
    private: std::vector<std::string> frame_name_;

        // Additional information
    private: long step = 0;
    private: bool verbose = false;
    private: common::Time prevUpdateTime;

        // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;

        /// \brief The load function is called by Gazebo when the plugin is
        /// inserted into simulation
        /// \param[in] _model A pointer to the model that this plugin is
        /// attached to.
        /// \param[in] _sdf A pointer to the plugin's SDF element.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
            {
                // // std::string file = "model://biorob_salamander/biorob_salamander.sdf";
                // std::string file = "/home/jonathan/.gazebo/models/biorob_salamander/biorob_salamander.sdf";
                // std::ifstream t(file);
                // std::string content(
                //     (std::istreambuf_iterator<char>(t)),
                //     (std::istreambuf_iterator<char>()));
                // std::cout << content << std::endl;

                // std::string name = this->GetFilename();
                // std::cout << name << std::endl;

                // if (_sdf->HasElement("gazebo_plugin_path"))
                //     std::cout << "Gazebo plugin path: " << _sdf->Get<std::string>("gazebo_plugin_path") << std::endl;
                // else
                //     std::cout << "gazebo_plugin_path not found in sdf" << std::endl;

                // Store the pointer to the model
                this->model = _model;

                // Load confirmation message
                std::cout
                    << "\nThe salamander plugin is attached to model["
                    << this->model->GetName()
                    << "]"
                    << std::endl;

                // Joints
                int joints_n = this->model->GetJointCount();
                std::vector<physics::JointPtr> joints_all = this->model->GetJoints();
                joints_names.resize(joints_n);
                int i = 0;
                std::cout << "List of joints:" << std::endl;
                std::shared_ptr<Joint> joint_ptr;
                for (auto &joint: joints_all) {
                    joints_names[i] = joint->GetName();
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
                    if (joint->HasType(gazebo::physics::Joint::FIXED_JOINT))
                        this->ft_sensors_names.push_back(joints_names[i]);
                    else
                    {
                        // Revolute joint

                        // PID position
                        name = "joints_" + joints_names[i] + "_pid_position_p";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->pid_position.SetPGain(value);
                        }
                        name = "joints_" + joints_names[i] + "_pid_position_i";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->pid_position.SetIGain(value);
                        }
                        name = "joints_" + joints_names[i] + "_pid_position_d";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->pid_position.SetDGain(value);
                        }

                        // PID velocity
                        name = "joints_" + joints_names[i] + "_pid_velocity_p";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->pid_velocity.SetPGain(value);
                        }
                        name = "joints_" + joints_names[i] + "_pid_velocity_i";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->pid_velocity.SetIGain(value);
                        }
                        name = "joints_" + joints_names[i] + "_pid_velocity_d";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->pid_velocity.SetDGain(value);
                        }

                        // oscillator options
                        name = "joints_" + joints_names[i] + "_type";
                        if(_sdf->HasElement(name))
                        {
                            value_str = _sdf->Get<std::string>(name);
                            std::cout << "    Setting " << name << " = " << value_str << std::endl;
                            this->joints[joints_names[i]]->oscillator.type = value_str;
                        }
                        name = "joints_" + joints_names[i] + "_amplitude";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->oscillator["amplitude"] = value;
                        }
                        name = "joints_" + joints_names[i] + "_frequency";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->oscillator["frequency"] = value;
                        }
                        name = "joints_" + joints_names[i] + "_phase";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->oscillator["phase"] = value;
                        }
                        name = "joints_" + joints_names[i] + "_bias";
                        if(_sdf->HasElement(name))
                        {
                            value = _sdf->Get<double>(name);
                            std::cout << "    Setting " << name << " = " << value << std::endl;
                            this->joints[joints_names[i]]->oscillator["bias"] = value;
                        }
                    }
                    i++;
                }
                std::cout << std::endl;

                // FT sensors
                this->ft_sensors.resize(this->ft_sensors_names.size());
                i = 0;
                for (auto &name: this->ft_sensors_names) {
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

                if(verbose)
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
                        if (verbose) {
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
