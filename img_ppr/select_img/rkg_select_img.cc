#include "rkg_select_img.h"


RKG_SELECT_IMG::RKG_SELECT_IMG(const char* config_json_path) :
image_table(), batch_proc_button("BATCH PROCESS"), 
main_box(Gtk::ORIENTATION_VERTICAL, 10), 
srcPATH(RKG_SELECT_IMG::read_config_json(config_json_path)["srcPATH"].asCString()), 
destPATH(RKG_SELECT_IMG::read_config_json(config_json_path)["destPATH"].asCString()),
selres_mngr(RKG_SELECT_IMG::read_config_json(config_json_path)["dbPATH"].asCString(), 
RKG_SELECT_IMG::read_config_json(config_json_path)["tableNAME"].asCString())
{
    //initialize member variables.
    Json::Value config_obj = RKG_SELECT_IMG::read_config_json(config_json_path);
    Json::Value heights = config_obj["resizeH"];
    Json::Value widths = config_obj["resizeW"];
    int len_Hs = std::size(heights);
    int len_Ws = std::size(widths);
    if (len_Hs != len_Ws)
        throw std::invalid_argument("The length of the height array does not equal to that of the width array.");
    this->num_img_size = len_Hs;
    
    this->dest_heights = new int[len_Hs];
    this->dest_widths = new int[len_Ws];
    for (int i=0; i<this->num_img_size; i++)
        {dest_heights[i] = heights[i].asInt(); dest_widths[i] = widths[i].asInt();}
    this->dirp = opendir(this->srcPATH.c_str());
    
    load_images_on_table(false);
    set_title("RKG_SELECT_IMAGES");
    set_border_width(10);
    add(main_box);
    batch_proc_button.signal_clicked().connect(sigc::mem_fun(*this, &RKG_SELECT_IMG::on_batch_proc_button_clicked));
    main_box.pack_start(image_table());
    main_box.pack_start(batch_proc_button);
    show_all_children();
}

RKG_SELECT_IMG::~RKG_SELECT_IMG()
{
    closedir(this->dirp);
    for (int i=0; i<RKG_NUM_ROWS; i++)
    {
        //for (int j=0; j<RKG_NUM_COLS; j++)
        //    delete src_file_names[i][j];
        delete src_file_names[i];
    }
    delete src_file_names;
}

Json::Value RKG_SELECT_IMG::read_config_json(const char* config_json_path)
{
    std::ifstream config_json_ifs(config_json_path);
    Json::Reader reader;
    Json::Value config_obj;

    reader.parse(config_json_ifs, config_obj);

    return config_obj;
    /*
    this->srcPATH = config_obj["srcPATH"].asString();
    this->destPATH = config_obj["destPATH"].asString();
    const char* db_path = config_obj["dbPATH"].asCString();
    const char* tbl_name = config_obj["tableNAME"].asCString();

    Json::Value heights = config_obj["resizeH"];
    Json::Value widths = config_obj["resizeW"];
    int len_Hs = std::size(heights);
    int len_Ws = std::size(widths);
    if (len_Hs != len_Ws)
        throw std::invalid_argument("The length of the height array does not equal to that of the width array.");
    this->num_img_size = len_Hs;
    
    this->dest_heights = new int[len_Hs];
    this->dest_widths = new int[len_Ws];
    for (int i=0; i<this->num_img_size; i++)
        {dest_heights[i] = heights[i].asInt(); dest_widths[i] = widths[i].asInt();}

    this->selres_mngr(db_path, tbl_name);
    this->dirp = opendir(this->srcPATH.c_str());
    */
}

std::string RKG_SELECT_IMG::generate_src_path_str(const char* src_file_name)
{
    boost::filesystem::path src_dir(this->srcPATH.c_str());
    boost::filesystem::path src_file(src_file_name);
    return std::string((src_dir / src_file).c_str());
}

std::string RKG_SELECT_IMG::generate_src_path_str(std::string src_file_name)
{
    return generate_src_path_str(src_file_name.c_str());
}

std::string RKG_SELECT_IMG::generate_dest_path_str(const char* dest_file_name, int width, int height)
{
    boost::filesystem::path dest_dir(this->destPATH.c_str());
    boost::filesystem::path size_dir((std::to_string(width) + std::string("x") + std::to_string(height)).c_str());
    DIR* destdir;
    destdir = opendir(destPATH.c_str());
    if (destdir == NULL)
        std::cout << "destdir does not exist." << std::endl;
    dest_dir = dest_dir / size_dir;
    boost::filesystem::path dest_file(dest_file_name);
    if ((mkdir(dest_dir.c_str(), 0777)) == -1) //(mkdir(dest_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if (errno != EEXIST)
        {
            std::cout << "error occured on making the directory " << dest_dir << "." << std::endl;
            throw std::runtime_error(strerror(errno));
        }
    }
    return std::string((dest_dir / dest_file).c_str());
}

std::string RKG_SELECT_IMG::generate_dest_path_str(std::string dest_file_name, int width, int height)
{
    return generate_dest_path_str(dest_file_name.c_str(), width, height);
}

bool RKG_SELECT_IMG::check_dupplicated(const char* src_file_name)
{
    bool check;
    bool total_check = true; // if all dest_paths determined as dupplicated, return dupplicated.
    //std::cout << src_file_name << " : " << selres_mngr.check_dupplicated(generate_src_path_str(src_file_name).c_str(), NULL) << std::endl;
    if (selres_mngr.check_dupplicated(generate_src_path_str(src_file_name).c_str(), NULL))
    {
        return true;
    }
    for (int i=0; i<num_img_size; i++)
    {
        int w = dest_widths[i];
        int h = dest_heights[i];
        std::string src_path_ = generate_src_path_str(src_file_name);
        std::string dest_path_ = generate_dest_path_str(src_file_name, w, h);
        check = this->selres_mngr.check_dupplicated(src_path_.c_str(),dest_path_.c_str());
        total_check = (total_check && check);
    }
    return total_check;
}

void RKG_SELECT_IMG::load_images_on_table(bool call_dupplicate_msg)
{
    struct dirent *directory;
    image_table.set_selects_disable();
    for (int i=0; i<RKG_NUM_ROWS; i++)
    {
        for (int j=0; j<RKG_NUM_COLS; j++)
        {
            this->image_table(i,j).clear_image();
            this->src_file_names[i][j] = std::string("");
            image_table(i,j).set_select(RADIO_BUTTON_ON_DISCARD);
            image_table(i,j)().hide();
        }
    }
    int i=0;
    int i_max = RKG_NUM_COLS*RKG_NUM_ROWS;
    while (i<i_max)
    {
        if ((directory = readdir(this->dirp)) != NULL)
        {
            if (!((std::string(directory->d_name) == std::string(".")) || (std::string(directory->d_name) == std::string(".."))))
            {
                if (!check_dupplicated(directory->d_name))
                {
                    int row = i / RKG_NUM_COLS;
                    int col = i % RKG_NUM_COLS;
                    this->image_table(row,col).set_image(Gdk::Pixbuf::create_from_file(
                    generate_src_path_str(directory->d_name)), IMG_BOX_W, IMG_BOX_H);
                    this->src_file_names[row][col] = std::string(directory->d_name);
                    i++;
                    image_table(row, col).set_select_bttn_enable();
                    image_table(row, col)().show_all();
                }
            }
        }
        else
        {
            if (call_dupplicate_msg)
                Gtk::MessageDialog(*this, "There no file exists to read anymore.").run();
            break;
        }
    }
}

void RKG_SELECT_IMG::on_batch_proc_button_clicked()
{
    this->image_table.set_selects_disable();
    for (int i=0; i<RKG_NUM_ROWS; i++)
    {
        for (int j=0; j<RKG_NUM_COLS; j++)
        {
            std::string buf_src = generate_src_path_str(src_file_names[i][j]);
            if (image_table(i,j).get_select()==RADIO_BUTTON_ON_SELECT)
            {
                for (int k=0; k<num_img_size; k++)
                {
                    std::string buf_dest = generate_dest_path_str(src_file_names[i][j], dest_widths[k], dest_heights[k]);
                    if (!(this->src_file_names[i][j]==std::string("")) && (!(this->selres_mngr.check_dupplicated(buf_src, buf_dest))))
                    {
                        image_table(i,j).save_image(generate_dest_path_str(this->src_file_names[i][j].c_str(), 
                        dest_widths[k], dest_heights[k]).c_str(), dest_widths[k], dest_heights[k]);
                        std::cout << "save " << buf_src << " as " << buf_dest << ";" << std::endl;
                        this->selres_mngr.insert_log(buf_src, buf_dest, RADIO_BUTTON_ON_SELECT, 
                        (std::to_string(dest_widths[k]) + "x" + std::to_string(dest_heights[k])).c_str());
                        std::cout << "write log for the image to store as " << buf_dest << ";" << std::endl << std::endl;
                    }
                }
            }
            else
            {
                this->selres_mngr.insert_log(buf_src.c_str(), NULL, RADIO_BUTTON_ON_DISCARD, "");
                std::cout << "write log for discarded image, " << buf_src << std::endl << std::endl;
            }
        }
    }
    load_images_on_table();
    this->image_table.set_selects_enable();
}