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

#include <salamander_kinematics.pb.h>
#include "parameters.hh"


class LogModelKinematics
{
public:
    LogModelKinematics(gazebo::common::Time time, gazebo::physics::ModelPtr model, sdf::ElementPtr sdf)
        {
            this->model = model;
            this->parameters = get_parameters(sdf);
            this->filename = getenv("HOME")+this->parameters["filename"].as<std::string>();

            // Links
            YAML::Node _links = this->parameters["links"];
            for(YAML::const_iterator it=_links.begin(); it!=_links.end(); ++it)
            {
                if (this->verbose)
                    std::cout
                        << "  - Link "
                        << it->first
                        << " to be logged at "
                        << it->second["frequency"]
                        << " [Hz]"
                        << std::endl;
                salamander::msgs::LinkKinematics* link_msg = this->log_msg.add_links();
                link_msg->set_name(it->first.as<std::string>());
                this->links.insert({it->first.as<std::string>(), link_msg});
                this->log_link(time, link_msg, this->model->GetLink(it->first.as<std::string>()));
            }
            

            // Joints
            YAML::Node _joints = this->parameters["joints"];
            for(YAML::const_iterator it=_joints.begin(); it!=_joints.end(); ++it)
            {
                if (this->verbose)
                    std::cout
                        << "  - Joint "
                        << it->first
                        << " to be logged at "
                        << it->second["frequency"]
                        << " [Hz]"
                        << std::endl;
                salamander::msgs::JointKinematics* joint_msg = this->log_msg.add_joints();
                joint_msg->set_name(it->first.as<std::string>());
                this->joints.insert({it->first.as<std::string>(), joint_msg});
                this->log_joint(time, joint_msg, this->model->GetJoint(it->first.as<std::string>()));
            }

            if (this->verbose)
                std::cout
                    << "Model logs will be saved to "
                    << this->filename
                    << " upon deletion of the model"
                    << std::endl;
        };
    virtual ~LogModelKinematics(){};

public:
    void log(gazebo::common::Time time)
        {
            // Links
            YAML::Node _links = this->parameters["links"];
            for(YAML::const_iterator it=_links.begin(); it!=_links.end(); ++it)
            {
                // Previous time
                salamander::msgs::LinkKinematics* link_kin = this->links[it->first.as<std::string>()];
                int size = link_kin->state().size();
                gazebo::common::Time t = Convert(link_kin->state(size-1).time());
                if (time.Double() - t.Double() >= 1./it->second["frequency"].as<double>())
                {
                    this->log_link(time, link_kin, this->model->GetLink(it->first.as<std::string>()));
                }
            }

            // Joints
            YAML::Node _joints = this->parameters["joints"];
            for(YAML::const_iterator it=_joints.begin(); it!=_joints.end(); ++it)
            {
                // Previous time
                salamander::msgs::JointKinematics* joint_kin = this->joints[it->first.as<std::string>()];
                int size = joint_kin->state().size();
                gazebo::common::Time t = Convert(joint_kin->state(size-1).time());
                if (time.Double() - t.Double() >= 1./it->second["frequency"].as<double>())
                {
                    this->log_joint(time, joint_kin, this->model->GetJoint(it->first.as<std::string>()));
                }
            }
        };

    void log_link(gazebo::common::Time time,salamander::msgs::LinkKinematics* link_kin, gazebo::physics::LinkPtr link)
        {
            // Memory allocation
            salamander::msgs::LinkState *msg = link_kin->add_state();
            // Time
            gazebo::msgs::Time *_time = new gazebo::msgs::Time;
            gazebo::msgs::Set(_time, time);
            msg->set_allocated_time(_time);
            // Pose
            gazebo::msgs::Pose *_pose = new gazebo::msgs::Pose;
            gazebo::msgs::Set(_pose, link->WorldPose());
            msg->set_allocated_pose(_pose);
            // Velocity
            // optional gazebo.msgs.Vector3d linear_velocity  = 3;
            gazebo::msgs::Vector3d *linear_velocity = new gazebo::msgs::Vector3d;
            gazebo::msgs::Set(linear_velocity, link->RelativeLinearVel());
            msg->set_allocated_linear_velocity(linear_velocity);
            // optional gazebo.msgs.Vector3d angular_velocity = 4;
            gazebo::msgs::Vector3d *angular_velocity = new gazebo::msgs::Vector3d;
            gazebo::msgs::Set(angular_velocity, link->RelativeAngularVel());
            msg->set_allocated_angular_velocity(angular_velocity);
        }

    void log_joint(gazebo::common::Time time,salamander::msgs::JointKinematics* joint_kin, gazebo::physics::JointPtr joint)
        {
            // Memory allocation
            salamander::msgs::JointState *msg = joint_kin->add_state();
            // Time
            gazebo::msgs::Time *_time = new gazebo::msgs::Time;
            gazebo::msgs::Set(_time, time);
            msg->set_allocated_time(_time);
            // Pose
            msg->set_position(joint->Position(0));
            msg->set_velocity(joint->GetVelocity(0));
        }

    void dump()
        {
            if (this->verbose)
                std::cout << "Logging data" << std::endl;
            // Serialise and store data
            std::string data;
            std::ofstream myfile;
            myfile.open(this->filename);
            this->log_msg.SerializeToString(&data);
            myfile << data;
            myfile.close();
            if (this->verbose)
                std::cout << "Logged data" << std::endl;
        };

public:
    bool verbose=true;

private:
    PluginParameters parameters;
    gazebo::physics::ModelPtr model;
    salamander::msgs::ModelKinematics log_msg;
    std::unordered_map<std::string, salamander::msgs::LinkKinematics*> links;
    std::unordered_map<std::string, salamander::msgs::JointKinematics*> joints;
    std::string filename;
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
                if (this->model_logs->verbose)
                    std::cout
                        << "Model "
                        << this->model->GetName()
                        << ": Logging in progress"
                        << std::endl;
                this->model_logs->dump();
                if (this->model_logs->verbose)
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
        physics::WorldPtr world_;
        LogModelKinematics* model_logs;

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

                // Save pointers
                this->world_ = this->model->GetWorld();

                // Logs
                this->model_logs = new LogModelKinematics(this->get_time(), _model, _sdf);

                // Load confirmation message
                if (this->model_logs->verbose)
                    std::cout
                        << "\nThe salamander links logging plugin is attached to model["
                        << this->model->GetName()
                        << "]"
                        << std::endl;

                // Listen to the update event. This event is broadcast every
                // simulation iteration.
                this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&LogKinematicsPlugin::OnUpdate, this));
            }

        // Called by the world update start event
        virtual void OnUpdate()
            {
                // Logging
                this->model_logs->log(this->get_time());
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
    };

// Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(LogKinematicsPlugin);

}

#endif
