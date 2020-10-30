#include "img_select_box.h"


const Gdk::InterpType IMG_SELECT_BOX::DEFAULT_INTERP_TYPE = Gdk::INTERP_BILINEAR;
const char* IMG_SELECT_BOX::DEFAULT_IMG_EXTENSION = "jpeg";


IMG_SELECT_BOX::IMG_SELECT_BOX() :
    WidgetContainer(Gtk::ORIENTATION_VERTICAL, 10),
    RadioButtons(Gtk::ORIENTATION_HORIZONTAL, 10),
    RB_Select("Save"),
    RB_Discard("Discard")
{
    init(Glib::RefPtr<Gdk::Pixbuf>(), 0, 0, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);
}

IMG_SELECT_BOX::IMG_SELECT_BOX(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf) : 
    WidgetContainer(Gtk::ORIENTATION_VERTICAL, 10),
    RadioButtons(Gtk::ORIENTATION_HORIZONTAL, 10),
    RB_Select("Save"),
    RB_Discard("Discard")
{
    init(pixbuf, 0, 0, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);
}

IMG_SELECT_BOX::IMG_SELECT_BOX(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int img_width, int img_height) : 
    WidgetContainer(Gtk::ORIENTATION_VERTICAL, 10),
    RadioButtons(Gtk::ORIENTATION_HORIZONTAL, 10),
    RB_Select("Save"),
    RB_Discard("Discard")
{
    init(pixbuf, img_width, img_height, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);
}

IMG_SELECT_BOX::~IMG_SELECT_BOX()
{}

void IMG_SELECT_BOX::init(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int img_width, int img_height, Gdk::InterpType interp_type)
{
    // initialize radio buttons.
    RB_Discard.join_group(RB_Select);
    RadioButtons.pack_start(RB_Select);
    RadioButtons.pack_start(RB_Discard);
    RB_Select.set_active();

    WidgetContainer.set_border_width(10);

    set_image(pixbuf, img_width, img_height, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);

    WidgetContainer.pack_start(IMG);
    WidgetContainer.pack_start(m_Separator);
    WidgetContainer.pack_start(RadioButtons);
}

void IMG_SELECT_BOX::set_image(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int dest_width, int dest_height, Gdk::InterpType interp_type)
{
    /*
    if (Orig_Pixbuf.get())
        Orig_Pixbuf.reset();
    */

    if (IMG.get_pixbuf())
        clear_image();

    Orig_Pixbuf = pixbuf;

    if (pixbuf.get())
    {
        if (dest_width==0)
            dest_width = pixbuf.get()->get_width();
        if (dest_height==0)
            dest_height = pixbuf.get()->get_height();
        if (this->img_width != dest_width || this->img_height != dest_height)
            IMG.set(scale_pixbuf(dest_width, dest_height, interp_type));
        else
            IMG.set(Orig_Pixbuf);
    }
    this->img_width = dest_width;
    this->img_height = dest_height;
    set_img_size_request(img_width, img_width);
}

void IMG_SELECT_BOX::set_image(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int dest_width, int dest_height)
{
    set_image(pixbuf, dest_width, dest_height, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);
}

void IMG_SELECT_BOX::set_img_size_request(int width, int height)
{
    IMG.set_size_request(width, height);
}

void IMG_SELECT_BOX::set_image(Gdk::Pixbuf*& pixbuf, int dest_width, int dest_height, Gdk::InterpType interp_type)
{
    set_image(Glib::RefPtr<Gdk::Pixbuf>(pixbuf), dest_width, dest_height, interp_type);
}

void IMG_SELECT_BOX::clear_image()
{
    IMG.clear();
    Orig_Pixbuf.reset();
    this->img_width = 0;
    this->img_height = 0;
}

Glib::RefPtr<Gdk::Pixbuf> IMG_SELECT_BOX::scale_pixbuf(int dest_width, int dest_height, Gdk::InterpType interp_type)
{
    return Orig_Pixbuf.get()->scale_simple(dest_width, dest_height, interp_type);
}

Glib::RefPtr<Gdk::Pixbuf> IMG_SELECT_BOX::scale_pixbuf(int dest_width, int dest_height)
{
    return scale_pixbuf(dest_width, dest_height, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);
}

void IMG_SELECT_BOX::save_image(const char* save_path, const char* type,int width, int height, Gdk::InterpType interp_type)
{
    Glib::RefPtr<Gdk::Pixbuf> buf(Orig_Pixbuf);
    int orig_w = buf.get()->get_width();
    int orig_h = buf.get()->get_height();
    if (width==0)
        width = orig_w;
    if (height==0)
        height = orig_h;

    if (!((width == buf->get_width()) || (height == buf->get_height())))
        buf = scale_pixbuf(width, height, interp_type);
    buf->save(std::string(save_path), Glib::ustring(type));
}

void IMG_SELECT_BOX::save_image(const char* save_path, const char* type,int width, int height)
{
    save_image(save_path, type, width, height, IMG_SELECT_BOX::DEFAULT_INTERP_TYPE);
}

void IMG_SELECT_BOX::save_image(const char* save_path, const char* type)
{
    save_image(save_path, type, 0, 0);
}

void IMG_SELECT_BOX::save_image(const char* save_path, int width, int height)
{
    save_image(save_path, IMG_SELECT_BOX::DEFAULT_IMG_EXTENSION, width, height);
}


int IMG_SELECT_BOX::get_select()
{
    if (RB_Select.get_active())
        return RADIO_BUTTON_ON_SELECT;
    else
        return RADIO_BUTTON_ON_DISCARD;
}

void IMG_SELECT_BOX::set_select_bttn_enable()
{
    RB_Select.set_sensitive(true);
    RB_Discard.set_sensitive(true);
    select_buttons_sensitive = true;
}

void IMG_SELECT_BOX::set_select_bttn_disable()
{
    RB_Select.set_sensitive(false);
    RB_Discard.set_sensitive(false);
    select_buttons_sensitive = false;
}

bool IMG_SELECT_BOX::get_select_bttn_sensitive()
{
    return select_buttons_sensitive;
}

void IMG_SELECT_BOX::set_select(int select)
{
    switch (select)
    {
        case RADIO_BUTTON_ON_SELECT:
            RB_Select.set_active();
            break;
        case RADIO_BUTTON_ON_DISCARD:
            RB_Select.set_active();
            break;
        default:
            break;
    }
}



