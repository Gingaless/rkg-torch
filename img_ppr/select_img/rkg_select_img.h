#ifndef RKG_SELECT_IMG_H
#define RKG_SELECT_IMG_H

#include "img_table.h"
#include "sql_selres.h"
#include <jsoncpp/json/json.h>
#include <gtkmm/window.h>
#include <gtkmm/button.h>
#include <gtkmm/box.h>
#include <gtkmm/messagedialog.h>
#include <dirent.h>
#include "sqlite3.h"
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <errno.h>
#include <iostream>
#include <dirent.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#define RKG_NUM_ROWS 3
#define RKG_NUM_COLS 4
#define IMG_BOX_H 128
#define IMG_BOX_W 128
#define FILES_REMAINED 1
#define NO_FILE 0


class RKG_SELECT_IMG : public Gtk::Window
{
    public:
        static Json::Value read_config_json(const char* config_json_path);
        RKG_SELECT_IMG(const char* config_json_path);
        virtual ~RKG_SELECT_IMG();

    protected:

        std::string generate_src_path_str(const char* src_file_name);
        std::string generate_src_path_str(std::string src_file_name);
        std::string generate_dest_path_str(const char* src_file_name, int width, int height);
        std::string generate_dest_path_str(std::string src_file_name, int width, int height);
        //const char** generate_dest_path_str_all(const char** src_file_name);
        void on_batch_proc_button_clicked();
        void load_images_on_table(bool call_dupplicate_msg = true);
        bool check_dupplicated(const char* src_file_name);


    RKG_SELRES_MANAGER selres_mngr;
    std::string srcPATH;
    std::string destPATH;
    int* dest_heights;
    int* dest_widths;
    std::string src_file_names[RKG_NUM_ROWS][RKG_NUM_COLS];
    int num_img_size;
    Gtk::Box main_box;
    IMG_TABLE<RKG_NUM_ROWS, RKG_NUM_COLS, IMG_BOX_W, IMG_BOX_H> image_table;
    Gtk::Button batch_proc_button;
    DIR* dirp;

};

#endif //RKG_SELECT_IMG_H
