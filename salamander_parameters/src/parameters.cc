#include "parameters.hh"


void print_yaml(YAML::Node node, std::string tab="") {
    for(YAML::const_iterator it=node.begin(); it!=node.end(); ++it) {
        std::cout << tab << it->first << ":";
        if (it->second.IsScalar())
            std::cout << " " << it->second << std::endl;
        else {
            std::cout << std::endl;
            print_yaml(it->second, tab+" ");
        }
    }
}

void PluginParameters::load(std::string filename, bool verbose) {
    std::string _filename = getenv("HOME")+filename;
    std::cout << "Loading " << _filename << std::endl;
    this->config = YAML::LoadFile(_filename);
    std::cout << _filename << " loaded" << std::endl;
    if (verbose)
    {
        std::cout << "Config parameters:" << std::endl;
        print_yaml(config);
    }
    std::cout << "Plugin parameters loaded" << std::endl;
}

YAML::Node PluginParameters::operator[](std::string name) {
    return this->config[name];
}

PluginParameters get_parameters(sdf::ElementPtr sdf, bool verbose)
{
    PluginParameters parameters;
    if (sdf->HasElement("config"))
    {
        std::string config_filename = sdf->Get<std::string>("config");
        if (verbose)
            std::cout
                << "    Config found: "
                << config_filename
                << std::endl;
        if (verbose)
            std::cout << "Loading parameters from " << config_filename << std::endl;
        parameters.load(config_filename);
    }
    else
    {
        std::cerr
            << "ERROR: config not found in plugin"
            << std::endl;
    }
    return parameters;
}
