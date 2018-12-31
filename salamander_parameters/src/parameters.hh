#ifndef _PARAMETERS_PLUGIN_HH_
#define _PARAMETERS_PLUGIN_HH_

#include <memory>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <math.h>
#include <yaml-cpp/yaml.h>


class PluginParameters
{
public:
    PluginParameters(){};
    virtual ~PluginParameters(){};

private:
    std::string filename;
    YAML::Node config;

public:
    void load(std::string filename, bool verbose=false);
    YAML::Node operator[](std::string name);

};

#endif
