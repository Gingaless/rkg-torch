#include "rkg_select_img.h"
#include <gtkmm/application.h>
#include <boost/stacktrace.hpp>

const char* config_json_path = "config.json";

int main(int argc, char *argv[])
{
    try
    {
        auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");
        RKG_SELECT_IMG rkg_sel_img(config_json_path);
        return app->run(rkg_sel_img);
    }
    catch (std::exception e)
    {
        std::cout << "error occured." << std::endl << std::endl;
        std::cout << boost::stacktrace::stacktrace() << std::endl;
    }
     
}