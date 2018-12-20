#ifndef _BENCHMARK_PLUGIN_HH_
#define _BENCHMARK_PLUGIN_HH_

#include <ignition/math/Pose3.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"


namespace gazebo
{
    class BenchmarkPlugin : public ModelPlugin
    {
    private:
        physics::ModelPtr model;
        physics::WorldPtr world_;
        event::ConnectionPtr updateConnection;

    public:
        BenchmarkPlugin(){};
        virtual ~BenchmarkPlugin(){
            std::cout << "Benchmark plugin: Closed" << std::endl;
        };

        void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
            {
                std::cout << "Benchmark plugin: Loading" << std::endl;
                this->model = _parent;
                this->world_ = this->model->GetWorld();
                this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&BenchmarkPlugin::OnUpdate, this));
                std::cout << "Benchmark plugin: Loaded" << std::endl;
            }

        virtual void OnUpdate()
            {
                // Time
#if GAZEBO_MAJOR_VERSION >= 8
                common::Time cur_time = this->world_->SimTime();
#else
                common::Time cur_time = this->world_->GetSimTime();
#endif
                double time = cur_time.sec + 1e-9*cur_time.nsec;

                if(time > 3)
                {
                    std::cout << "Benchmark could be computed now" << std::endl;
                    this->model->WorldPose();
                    std::raise(SIGINT);
                }
            }
    };

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(BenchmarkPlugin)
}

#endif
