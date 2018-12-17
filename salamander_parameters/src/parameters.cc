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

void PluginParameters::load(std::string filename) {
    std::string _filename = getenv("HOME")+filename;
    std::cout << "Loading " << _filename << std::endl;
    this->config = YAML::LoadFile(_filename);
    std::cout << _filename << " loaded" << std::endl;
    std::cout << "Config parameters:" << std::endl;
    print_yaml(config);
    std::cout << "Plugin parameters loaded" << std::endl;
}

YAML::Node PluginParameters::operator[](std::string name) {
    return this->config[name];
}
