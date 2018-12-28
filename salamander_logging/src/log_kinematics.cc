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


class LogLink
{
public:
    LogLink(std::string name, double frequency){
        data.set_name(name);
        this->ifreq = 1./frequency;
    };
    virtual ~LogLink(){};

private:
    double time_tol = 1e-6;
    salamander::msgs::LinkKinematics data;
    double ifreq;
    double time_last_log = -1;

public:
    void log(gazebo::common::Time time, gazebo::physics::LinkPtr link){
        if (this->check_if_log(time.sec+1e-9*time.nsec))
        {
            this->_log(time, link);
        }
        return;
    }

    salamander::msgs::LinkKinematics get_logs() {
        return this->data;
    }

private:
    bool check_if_log(double time) {
        bool result = ((time - time_last_log) - (this->ifreq) >= -time_tol) || (time < time_tol);
        if (result)
            time_last_log = time;
        return result;
    }

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
            // Linear_velocity
            // Angular_velocity
            
            // // Other
//             salamander::msgs::Vector3 *pos = new salamander::msgs::Vector3;
//             salamander::msgs::Rotation *rot = new salamander::msgs::Rotation;
//             salamander::msgs::Vector3 *rotx = new salamander::msgs::Vector3;
//             salamander::msgs::Vector3 *roty = new salamander::msgs::Vector3;
//             salamander::msgs::Vector3 *rotz = new salamander::msgs::Vector3;
//             salamander::msgs::Vector3 *lin_vel = new salamander::msgs::Vector3;
//             salamander::msgs::Vector3 *ang_vel = new salamander::msgs::Vector3;
//             // Variables
// #if GAZEBO_MAJOR_VERSION >= 8
//             ignition::math::Pose3d linkworldpose = link->WorldPose();
//             ignition::math::Vector3d linkworldpos = linkworldpose.Pos();
//             ignition::math::Quaterniond quat = linkworldpose.Rot();
//             ignition::math::Vector3d axis;
//             double angle;
//             quat.ToAxis(axis, angle);
//             ignition::math::Matrix3d linkworldrot;
//             linkworldrot(0, 0) = axis[0];
//             linkworldrot(0, 1) = axis[1];
//             linkworldrot(0, 2) = axis[2];
//             linkworldrot(1, 0) = angle;
//             ignition::math::Vector3d linkworldlinearvel = link->WorldLinearVel();
//             ignition::math::Vector3d linkworldangularvel = link->WorldAngularVel();
//             // Orientation
//             rotx->set_x(linkworldrot(0, 0));
//             rotx->set_y(linkworldrot(0, 1));
//             rotx->set_z(linkworldrot(0, 2));
//             rot->set_allocated_x(rotx);
//             roty->set_x(linkworldrot(1, 0));
//             roty->set_y(linkworldrot(1, 1));
//             roty->set_z(linkworldrot(1, 2));
//             rot->set_allocated_y(roty);
//             rotz->set_x(linkworldrot(2, 0));
//             rotz->set_y(linkworldrot(2, 1));
//             rotz->set_z(linkworldrot(2, 2));
//             rot->set_allocated_z(rotz);
//             msg->set_allocated_rot(rot);
// #else
//             ignition::math::Pose3d linkworldpose = link->GetWorldPose();
//             ignition::math::Vector3d linkworldpos = linkworldpose.pos;
//             ignition::math::Matrix3d linkworldrot = linkworldpose.rot.GetAsMatrix3();
//             ignition::math::Vector3d linkworldlinearvel = link->GetWorldLinearVel();
//             ignition::math::Vector3d linkworldangularvel = link->GetWorldAngularVel();
//             // Orientation
//             rotx->set_x(linkworldrot[0][0]);
//             rotx->set_y(linkworldrot[0][1]);
//             rotx->set_z(linkworldrot[0][2]);
//             rot->set_allocated_x(rotx);
//             roty->set_x(linkworldrot[1][0]);
//             roty->set_y(linkworldrot[1][1]);
//             roty->set_z(linkworldrot[1][2]);
//             rot->set_allocated_y(roty);
//             rotz->set_x(linkworldrot[2][0]);
//             rotz->set_y(linkworldrot[2][1]);
//             rotz->set_z(linkworldrot[2][2]);
//             rot->set_allocated_z(rotz);
//             msg->set_allocated_rot(rot);
// #endif
//             // Time
//             msg->set_sec(time.sec);
//             msg->set_nsec(time.nsec);
//             // Position
//             pos->set_x(linkworldpos[0]);
//             pos->set_y(linkworldpos[1]);
//             pos->set_z(linkworldpos[2]);
//             msg->set_allocated_pos(pos);
//             // Linear velocity
//             lin_vel->set_x(linkworldlinearvel[0]);
//             lin_vel->set_y(linkworldlinearvel[1]);
//             lin_vel->set_z(linkworldlinearvel[2]);
//             msg->set_allocated_linvel(lin_vel);
//             // Angular velocity
//             ang_vel->set_x(linkworldangularvel[0]);
//             ang_vel->set_y(linkworldangularvel[1]);
//             ang_vel->set_z(linkworldangularvel[2]);
//             msg->set_allocated_angvel(ang_vel);
        }
};


class LogParameters
{
public:
    LogParameters(){};
    virtual ~LogParameters(){};

public:
    std::unordered_map<std::string, LogLink> links;

private:
    std::string filename;

public:
    void parse_yaml(std::string filename) {
        std::string _filename = getenv("HOME")+filename;
        std::cout << "Loading " << _filename << std::endl;
        YAML::Node config = YAML::LoadFile(_filename);
        std::cout << _filename << " loaded" << std::endl;
        YAML::Node _links = config["links"];
        std::cout << "Links to log:" << std::endl;
        for(YAML::const_iterator it=_links.begin(); it!=_links.end(); ++it) {
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
        std::cout
            << "Links logs will be saved to "
            << this->filename
            << " upon deletion of the model"
            << std::endl;
        return;
    }

    void dump() {
        std::cout << "Logging data" << std::endl;
        salamander::msgs::ModelKinematics links_logs;
        salamander::msgs::LinkKinematics *link_logs_ptr;
        for (auto &link: this->links)
        {
            link_logs_ptr = links_logs.add_links();
            link_logs_ptr->MergeFrom(link.second.get_logs());
        }
        // Serialise and store data
        std::string data;
        std::ofstream myfile;
        myfile.open(getenv("HOME")+this->filename);
        links_logs.SerializeToString(&data);
        myfile << data;
        myfile.close();
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
                std::cout
                    << "Model "
                    << this->model->GetName()
                    << ": Logging plugin deleted"
                    << std::endl;
                this->links_logs.dump();
                return;
            }

    private:
        // Model information
        physics::ModelPtr model;
        std::vector<std::string> joints_names;
        std::unordered_map<std::string, physics::LinkPtr> links;
        physics::WorldPtr world_;

        // Additional information
        long step = 0;
        common::Time prevUpdateTime;

        // Logging
        LogParameters links_logs;

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
                std::cout
                    << "\nThe salamander links logging plugin is attached to model["
                    << this->model->GetName()
                    << "]"
                    << std::endl;

                // Get all links
                this->load_links(_model);

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
                std::cout << "Links found:";
                for (auto &link: links_all)
                {
                    std::cout << std::endl << "  " << link->GetName();
                    this->links.insert({link->GetName(), link});
                }
                std::cout << std::endl;
                return;
            }

        void load_sdf(sdf::ElementPtr _sdf)
            {
                std::cout << "SDF parameters:" << std::endl;
                std::string parameter = "config";
                std::string filename = "";
                std::string value;
                if(_sdf->HasElement(parameter))
                {
                    value = _sdf->Get<std::string>(parameter);
                    std::cout << "    Setting " << parameter << " = " << value << std::endl;
                    filename = value;
                    this->links_logs.parse_yaml(filename);
                }
                return;
            }

        void log(common::Time time)
            {
                for (auto &link_map: this->links_logs.links)
                {
                    link_map.second.log(time, this->links[link_map.first]);
                }
            }
    };

// Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(LogKinematicsPlugin);

}

#endif
