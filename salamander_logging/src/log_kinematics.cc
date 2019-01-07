#ifndef _LOG_KINEMATICS_PLUGIN_HH_
#define _LOG_KINEMATICS_PLUGIN_HH_

#include <memory>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <math.h>
#include <yaml-cpp/yaml.h>

#include <log_kinematics.pb.h>
// #include <salamander-msgs/log_kinematics.pb.h>


template <class msgType, class entityType>
class LogEntity
{
protected:
    msgType data;
    double time_tol = 1e-6;
    double ifreq;
    double time_last_log = -1;

public:
    LogEntity(std::string name, double frequency){
        data.set_name(name);
        this->ifreq = 1./frequency;
    };
    virtual ~LogEntity(){};

public:
    void log(gazebo::common::Time time, entityType entity){
        if (this->check_if_log(time.sec+1e-9*time.nsec))
        {
            this->_log(time, entity);
        }
        return;
    }

    msgType get_logs() {
        return this->data;
    }

private:
    bool check_if_log(double time) {
        bool result = ((time - time_last_log) - (this->ifreq) >= -time_tol) || (time < time_tol);
        if (result)
            time_last_log = time;
        return result;
    }

    virtual void _log(gazebo::common::Time time, entityType entity) = 0;
};


class LogLink:
    public LogEntity<
    salamander::msgs::LinkKinematics,
    gazebo::physics::LinkPtr>
{
public:
    LogLink(std::string name, double frequency): LogEntity(name, frequency){};
    virtual ~LogLink(){};

private:

    void _log(gazebo::common::Time time, gazebo::physics::LinkPtr link)
        {
            // Memory allocation
            salamander::msgs::LinkState *msg = this->data.add_state();
            // Time
            gazebo::msgs::Time *_time = new gazebo::msgs::Time;
            gazebo::msgs::Set(_time, time);
            msg->set_allocated_time(_time);
            // Pose
            gazebo::msgs::Pose *_pose = new gazebo::msgs::Pose;
            gazebo::msgs::Set(_pose, link->WorldPose());
            msg->set_allocated_pose(_pose);
        }
};


class LogJoint:
    public LogEntity<
    salamander::msgs::JointKinematics,
    gazebo::physics::JointPtr>
{
public:
    LogJoint(std::string name, double frequency): LogEntity(name, frequency){};
    virtual ~LogJoint(){};

private:

    void _log(gazebo::common::Time time, gazebo::physics::JointPtr joint)
        {
            // Memory allocation
            salamander::msgs::JointState *msg = this->data.add_state();
            // Time
            gazebo::msgs::Time *_time = new gazebo::msgs::Time;
            gazebo::msgs::Set(_time, time);
            msg->set_allocated_time(_time);
            // Pose
            msg->set_position(joint->Position(0));
            msg->set_velocity(joint->GetVelocity(0));
        }
};


class LogParameters
{
public:
    LogParameters(){};
    virtual ~LogParameters(){};

public:
    std::unordered_map<std::string, LogLink> links;
    std::unordered_map<std::string, LogJoint> joints;
    bool verbose = true;

private:
    std::string filename;

public:
    void parse_yaml(std::string filename) {
        std::string _filename = getenv("HOME")+filename;
        if (this->verbose)
            std::cout << "Loading " << _filename << std::endl;
        YAML::Node config = YAML::LoadFile(_filename);
        if (this->verbose)
            std::cout << _filename << " loaded" << std::endl;
        // Links
        YAML::Node _links = config["links"];
        if (this->verbose)
            std::cout << "Links to log:" << std::endl;
        for(YAML::const_iterator it=_links.begin(); it!=_links.end(); ++it) {
            if (this->verbose)
                std::cout
                    << "  - Link "
                    << it->first
                    << " to be logged at "
                    << it->second["frequency"]
                    << " [Hz]"
                    << std::endl;
            LogLink log(it->first.as<std::string>(), it->second["frequency"].as<double>());
            this->links.insert({it->first.as<std::string>(), log});
        }
        this->filename = config["filename"].as<std::string>();
        if (this->verbose)
            std::cout
                << "Links logs will be saved to "
                << this->filename
                << " upon deletion of the model"
                << std::endl;
        // Joints
        YAML::Node _joints = config["joints"];
        if (this->verbose)
            std::cout << "Joints to log:" << std::endl;
        for(YAML::const_iterator it=_joints.begin(); it!=_joints.end(); ++it) {
            if (this->verbose)
                std::cout
                    << "  - Joint "
                    << it->first
                    << " to be logged at "
                    << it->second["frequency"]
                    << " [Hz]"
                    << std::endl;
            LogJoint log(it->first.as<std::string>(), it->second["frequency"].as<double>());
            this->joints.insert({it->first.as<std::string>(), log});
        }
        this->filename = config["filename"].as<std::string>();
        if (this->verbose)
            std::cout
                << "Joints logs will be saved to "
                << this->filename
                << " upon deletion of the model"
                << std::endl;
        return;
    }

    void dump() {
        if (this->verbose)
            std::cout << "Logging data" << std::endl;
        salamander::msgs::ModelKinematics model_logs;
        salamander::msgs::LinkKinematics *link_logs_ptr;
        for (auto &link: this->links)
        {
            link_logs_ptr = model_logs.add_links();
            link_logs_ptr->MergeFrom(link.second.get_logs());
        }
        salamander::msgs::JointKinematics *joint_logs_ptr;
        for (auto &joint: this->joints)
        {
            joint_logs_ptr = model_logs.add_joints();
            joint_logs_ptr->MergeFrom(joint.second.get_logs());
        }
        // Serialise and store data
        std::string data;
        std::ofstream myfile;
        myfile.open(getenv("HOME")+this->filename);
        model_logs.SerializeToString(&data);
        myfile << data;
        myfile.close();
        if (this->verbose)
            std::cout << "Logged data" << std::endl;
    }

};


namespace gazebo
{
    /// \brief A plugin to control a Salamander sensor.
    class LogKinematicsPlugin : public ModelPlugin
    {
        /// \brief Constructor
    public:
        LogKinematicsPlugin() {
            // Verify that the version of the library that we linked against is
            // compatible with the version of the headers we compiled against.
            GOOGLE_PROTOBUF_VERIFY_VERSION;
        };
        ~LogKinematicsPlugin()
            {
                // Deletion message
                if (this->model_logs.verbose)
                    std::cout
                        << "Model "
                        << this->model->GetName()
                        << ": Logging in progress"
                        << std::endl;
                this->model_logs.dump();
                if (this->model_logs.verbose)
                    std::cout
                        << "Model "
                        << this->model->GetName()
                        << ": Logging plugin deleted"
                        << std::endl;
                return;
            }

    private:
        // Model information
        physics::ModelPtr model;
        std::vector<std::string> joints_names;
        std::unordered_map<std::string, physics::LinkPtr> links;
        std::unordered_map<std::string, physics::JointPtr> joints;
        physics::WorldPtr world_;

        // Additional information
        long step = 0;
        common::Time prevUpdateTime;

        // Logging
        LogParameters model_logs;

        // Pointer to the update event connection
        event::ConnectionPtr updateConnection;

        /// \brief The load function is called by Gazebo when the plugin is
        /// inserted into simulation
        /// \param[in] _model A pointer to the model that this plugin is
        /// attached to.
        /// \param[in] _sdf A pointer to the plugin's SDF element.
    public:
        virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
            {
                // Store the pointer to the model
                this->model = _model;

                // Load confirmation message
                if (this->model_logs.verbose)
                    std::cout
                        << "\nThe salamander links logging plugin is attached to model["
                        << this->model->GetName()
                        << "]"
                        << std::endl;

                // Get all links
                this->load_links(_model);

                // Get all joints
                this->load_joints(_model);

                // SDF
                this->load_sdf(_sdf);

                // Save pointers
                this->world_ = this->model->GetWorld();

                this->log(this->get_time());

                // Listen to the update event. This event is broadcast every
                // simulation iteration.
                this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&LogKinematicsPlugin::OnUpdate, this));
            }

        // Called by the world update start event
        virtual void OnUpdate()
            {
                // Time
                common::Time cur_time = this->get_time();
                common::Time stepTime = cur_time - this->prevUpdateTime;
                this->prevUpdateTime = cur_time;

                // Logging
                this->log(cur_time);

                // Iteration
                this->step++;
            }

    private:
        common::Time get_time()
            {
#if GAZEBO_MAJOR_VERSION >= 8
                common::Time cur_time = this->world_->SimTime();
#else
                common::Time cur_time = this->world_->GetSimTime();
#endif
                return cur_time;
            }
        
        void load_links(physics::ModelPtr _model)
            {
                std::vector<physics::LinkPtr> links_all = _model->GetLinks();
                if (this->model_logs.verbose)
                    std::cout << "Links found:";
                for (auto &link: links_all)
                {
                    if (this->model_logs.verbose)
                        std::cout << std::endl << "  " << link->GetName();
                    this->links.insert({link->GetName(), link});
                }
                if (this->model_logs.verbose)
                    std::cout << std::endl;
                return;
            }

        void load_joints(physics::ModelPtr _model)
            {
                std::vector<physics::JointPtr> joints_all = _model->GetJoints();
                if (this->model_logs.verbose)
                    std::cout << "Joints found:";
                for (auto &joint: joints_all)
                {
                    if (this->model_logs.verbose)
                        std::cout << std::endl << "  " << joint->GetName();
                    this->joints.insert({joint->GetName(), joint});
                }
                if (this->model_logs.verbose)
                    std::cout << std::endl;
                return;
            }

        void load_sdf(sdf::ElementPtr _sdf)
            {
                if (this->model_logs.verbose)
                    std::cout << "SDF parameters:" << std::endl;
                std::string parameter = "config";
                std::string filename = "";
                std::string value;
                if(_sdf->HasElement(parameter))
                {
                    value = _sdf->Get<std::string>(parameter);
                    if (this->model_logs.verbose)
                        std::cout << "    Setting " << parameter << " = " << value << std::endl;
                    filename = value;
                    this->model_logs.parse_yaml(filename);
                }
                return;
            }

        void log(common::Time time)
            {
                for (auto &link_map: this->model_logs.links)
                {
                    link_map.second.log(time, this->links[link_map.first]);
                }
                for (auto &joint_map: this->model_logs.joints)
                {
                    joint_map.second.log(time, this->joints[joint_map.first]);
                }
            }
    };

// Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(LogKinematicsPlugin);

}

#endif
