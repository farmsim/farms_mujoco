#ifndef _VISOUCS_SWIMMING_PLUGIN_HH_
#define _VISOUCS_SWIMMING_PLUGIN_HH_

#include <memory>
#include <unordered_map>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <math.h>


class ViscousForcesVisuals
{
public:
    ViscousForcesVisuals(gazebo::physics::ModelPtr model){
        this->model = model;
        this->node = gazebo::transport::NodePtr(new gazebo::transport::Node());
#if GAZEBO_MAJOR_VERSION >= 8
        this->node->Init(model->GetWorld()->Name());
#else
        this->node->Init(model->GetWorld()->GetName());
#endif
        this->visPub = this->node->Advertise<gazebo::msgs::Visual>("~/visual", 10);
    };
    virtual ~ViscousForcesVisuals(){};

private:
    gazebo::physics::ModelPtr model;
    gazebo::transport::NodePtr node;
    gazebo::transport::PublisherPtr visPub;
    gazebo::msgs::Visual visualMsg;
    gazebo::msgs::Geometry *geomMsg;
    gazebo::msgs::Material *material;
    gazebo::msgs::Color* diffuse;

public:
    void update(std::string name, ignition::math::Pose3d pose)
        {
            // Set the visual's name. This should be unique.
            this->visualMsg.set_name(name);

            // Set the visual's parent. This visual will be attached to the parent
            this->visualMsg.set_parent_name(this->model->GetScopedName());

            // Create a cylinder
            this->geomMsg = this->visualMsg.mutable_geometry();
            this->geomMsg->set_type(gazebo::msgs::Geometry::CYLINDER);
            this->geomMsg->mutable_cylinder()->set_radius(0.005);
            this->geomMsg->mutable_cylinder()->set_length(0.1);

            // Set the material to be bright red
            // if ((!this->visualMsg.has_material()) || visualMsg.mutable_material() == NULL) {
            //     gazebo::msgs::Material *materialMsg = new gazebo::msgs::Material;
            //     this->diffuse = new gazebo::msgs::Color;
            //     visualMsg.set_allocated_material(materialMsg);
            // }
            this->diffuse = new gazebo::msgs::Color;
            this->material = this->visualMsg.mutable_material();
            if (this->material->has_diffuse())
            {
                this->material->clear_diffuse();
            }
            this->diffuse->set_r(1);
            this->diffuse->set_g(0);
            this->diffuse->set_b(0);
            this->diffuse->set_a(1);
            this->material->set_allocated_diffuse(diffuse);

            // Set the pose of the visual relative to its parent
            gazebo::msgs::Set(this->visualMsg.mutable_pose(), pose);

            // Don't cast shadows
            this->visualMsg.set_cast_shadows(false);

            // Publish
            this->visPub->Publish(this->visualMsg);
        }
};


namespace gazebo
{
    /// \brief A plugin to control a Salamander sensor.
    class ViscousSwimmingPlugin : public ModelPlugin
    {
        /// \brief Constructor
    public:
        ViscousSwimmingPlugin() {};
        ~ViscousSwimmingPlugin()
            {
                // Deletion message
                std::cout
                    << "Model "
                    << this->model->GetName()
                    << ": Viscous swimming plugin deleted"
                    << std::endl;
                return;
            }

    private:
        // Model information
        physics::ModelPtr model;
        std::vector<std::string> joints_names;
        std::unordered_map<std::string, physics::LinkPtr> links;
        std::vector<std::string> ft_sensors_names;
        // private: std::vector<FT_Sensor> ft_sensors;
        physics::WorldPtr world_;
        std::vector<std::string> frame_name_;

        // Additional information
        long step = 0;
        bool verbose = false;
        common::Time prevUpdateTime;

        // Visual
        ViscousForcesVisuals* visuals;

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
                this->visuals = new ViscousForcesVisuals(this->model);

                // Load confirmation message
                std::cout
                    << "\nThe salamander plugin is attached to model["
                    << this->model->GetName()
                    << "]"
                    << std::endl;

                // Get all links
                std::vector<physics::LinkPtr> links_all = this->model->GetLinks();
                std::cout << "Links found:";
                for (auto &link: links_all)
                {
                    std::cout << std::endl << "  " << link->GetName();
                    this->links.insert({link->GetName(), link});
                }
                std::cout << std::endl;

                // Save pointers
                this->world_ = this->model->GetWorld();

                // Listen to the update event. This event is broadcast every
                // simulation iteration.
                this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&ViscousSwimmingPlugin::OnUpdate, this));
            }

        // Called by the world update start event
        virtual void OnUpdate()
            {
                // Time
#if GAZEBO_MAJOR_VERSION >= 8
                common::Time cur_time = this->world_->SimTime();
#else
                common::Time cur_time = this->world_->GetSimTime();
#endif
                common::Time stepTime = cur_time - this->prevUpdateTime;
                this->prevUpdateTime = cur_time;

                // Viscous force
                ignition::math::Vector3<double> vel, force_0, force_1, swim_force, com_pos;
                double mass;
                for (auto &link_map: this->links)
                {
#if GAZEBO_MAJOR_VERSION >= 8
                mass = link_map.second->GetInertial()->Mass();
#else
                mass = link_map.second->GetInertial()->GetMass();
#endif

                    if (link_map.first.find("body") != std::string::npos && mass > 1e-2)
                    {
#if GAZEBO_MAJOR_VERSION >= 8
                        force_0 = link_map.second->WorldForce();
                        com_pos = link_map.second->GetInertial()->Pose().Pos();
                        vel = link_map.second->RelativeLinearVel();
#else
                        force_0 = link_map.second->GetWorldForce();
                        com_pos = link_map.second->GetInertial()->GetPose().pos;
                        vel = link_map.second->GetRelativeLinearVel();
#endif
                        // vel = link_map.second->GetWorldLinearVel(com_pos);
                        swim_force.Set(
                            -1e-1*copysign(1.0, vel[0])*vel[0]*vel[0],  // /mass
                            -3e-0*copysign(1.0, vel[1])*vel[1]*vel[1],
                            -3e-0*copysign(1.0, vel[2])*vel[2]*vel[2]);
                        link_map.second->AddRelativeForce(swim_force);
                        // link_map.second->AddForceAtRelativePosition(swim_force, com_pos);
#if GAZEBO_MAJOR_VERSION >= 8
                        force_1 = link_map.second->WorldForce();
#else
                        force_1 = link_map.second->GetWorldForce();
#endif

                        // Information
                        if (link_map.first == "link_body_11" && this->verbose)
                        {
                            std::cout << "\n" << link_map.first << ":" << std::endl;
                            std::cout << "  Velocity:\n" << vel << std::endl;
                            std::cout << "  com_pos:\n" << com_pos << std::endl;
                            std::cout << "  Force before:\n" << force_0 << std::endl;
                            std::cout << "  Force after:\n" << force_1 << std::endl;
                        }

                        // // Display
                        // ignition::math::Pose3d vis_pos;
                        // vis_pos.Set(
                        //     link_map.second->GetWorldCoGPose().pos[0],
                        //     link_map.second->GetWorldCoGPose().pos[1],
                        //     link_map.second->GetWorldCoGPose().pos[2],
                        //     0, 0, 0);
                        // this->visuals->update(
                        //     this->model->GetName()+"_"+link_map.first+"_forces",
                        //     vis_pos);
                    }
                }

                if(verbose)
                {
                    std::cout
                        << "The salamander viscous force plugin was called at step "
                        << this->step
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
                this->step++;
            }
    };

// Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
    GZ_REGISTER_MODEL_PLUGIN(ViscousSwimmingPlugin);

}

#endif
