#include <ignition/math/Pose3.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"

#include "mpi.h"

#define BUFFER_SIZE 100000


namespace gazebo
{
    class EvolutionPlugin : public WorldPlugin
    {
    private:
        physics::WorldPtr world_;
        event::ConnectionPtr updateConnection;
        bool verbose = false;

        // MPI
        int my_rank;
        int numprocs;
        MPI_Request req_recv;
        int flag;
        MPI_Status status;
        int count;
        char buffer[BUFFER_SIZE];
        char reply[BUFFER_SIZE];
        int source = 0;
        int tag_individual = 1;
        std::string message;
        int messages_received = 0;

        // Simulation
        double spawn_time = -1;

        int step = 0;

    public:
        EvolutionPlugin()
            {
                // MPI initializations
                int argc = 0;
                char** argv = NULL;
                MPI_Init (&argc, &argv);
                MPI_Comm_size (MPI_COMM_WORLD, &this->numprocs);
                MPI_Comm_rank (MPI_COMM_WORLD, &this->my_rank);
                double time_start = MPI_Wtime();
                std::cout << "Hello World, my rank is " << my_rank <<" "<< MPI_Wtime() - time_start << std::endl;

                // // req_recv = MPI_COMM_WORLD.irecv(dest=0, tag=1);
                // // this->buffer = new char[BUFFER_SIZE];
                // for (int i = 0; i < BUFFER_SIZE; ++i) {
                //     // this->buffer[i] = "a";
                //     strcpy(this->buffer, "a");
                // }

                MPI_Irecv(
                    this->buffer,
                    BUFFER_SIZE,
                    MPI_CHAR,
                    this->source,
                    this->tag_individual,
                    MPI_COMM_WORLD,
                    &this->req_recv);
                std::cout << "Initialisation complete" << std::endl;
                return;
            };

        virtual ~EvolutionPlugin()
            {
                // End MPI
                MPI_Finalize ();
                return;
            };

        void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
            {
                std::cout << "Evolution plugin loaded" << std::endl;

                this->world_ = _parent;

                // Listen to the update event. This event is broadcast every
                // simulation iteration.
                this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                    std::bind(&EvolutionPlugin::OnUpdate, this));
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
                if (this->verbose)
                {
                    if (this->step%1000 < 1)
                    {
                        std::cout
                            << "EVOLUTION PLUGIN: Time is "
                            << time
                            << " [s]"
                            << std::endl;
                    }
                    this->step++;
                    if (cur_time.sec >= 2){
                        std::cout
                            << "Resetting at iteration "
                            << this->step
                            << " (Time = "
                            << time
                            << "[s])"
                            << std::endl;
                        this->world_->Reset();
                    }
                }
                // std::cout << "About to receive message" << std::endl;
                // MPI_Request_get_status(*(this->req_recv), &this->flag, &this->status);
                MPI_Request_get_status(this->req_recv, &this->flag, &this->status);
                // std::cout << "Message received" << std::endl;
                MPI_Get_count(&this->status, MPI_CHAR, &this->count);
                if (this->count != 0)  // Message received
                {
                    this->message = std::string(this->buffer, this->count);
                    std::cout << "Status count: " << this->count << std::endl;
                    std::cout
                        << "Island: Message received: "
                        << this->message
                        << std::endl;
                    this->count = 0;
                    this->status = MPI_Status();
                    MPI_Request_free(&this->req_recv);
                    MPI_Irecv(
                        this->buffer,
                        BUFFER_SIZE,
                        MPI_CHAR,
                        this->source,
                        this->tag_individual,
                        MPI_COMM_WORLD,
                        &this->req_recv);

                    this->messages_received += 1;

                    // The filename must be in the GAZEBO_MODEL_PATH environment variable.
                    this->world_->InsertModelFile("model://"+this->message);
                    this->spawn_time = cur_time.Double();
                }

                if(this->spawn_time >= 0 && cur_time.Double() - this->spawn_time > 3)
                {
                    int dest = 0;
                    int tag = 1;
                    int result;
                    std::string answer = "I have correctly received message "+std::to_string(this->messages_received)+":\n" + this->message;
                    memcpy(this->reply, answer.data(), answer.length());
                    std::cout << "Island: Sending message back" << std::endl;
                    result = MPI_Send(
                        this->reply,
                        answer.length(),
                        MPI_CHAR,
                        dest,
                        tag,
                        MPI_COMM_WORLD);
                    std::cout << "Island: Message sent" << std::endl;
                    this->spawn_time = -1;
                }
            }
    };

    // Register this plugin with the simulator
    GZ_REGISTER_WORLD_PLUGIN(EvolutionPlugin)
}
