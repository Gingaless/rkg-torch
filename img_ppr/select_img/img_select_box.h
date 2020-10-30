#ifndef RKG_IMG_SELECT_BOX_H
#define RKG_IMG_SELECT_BOX_H

#define RADIO_BUTTON_ON_SELECT 1
#define RADIO_BUTTON_ON_DISCARD 0


#include <gtkmm/box.h>
#include <gtkmm/radiobutton.h>
#include <gtkmm/separator.h>
#include <gtkmm/image.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

/*
Gdk::InterpType { 
  Gdk::INTERP_TILES, = 1
  Gdk::INTERP_BILINEAR, = 2
  Gdk::INTERP_HYPER = 3
}
*/

class IMG_SELECT_BOX
{
    public:
        IMG_SELECT_BOX(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int img_width, int img_height, const Gdk::InterpType& interp_type);
        IMG_SELECT_BOX(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int img_width, int img_height);
        IMG_SELECT_BOX(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf);
        IMG_SELECT_BOX();
        //IMG_SELECT_BOX(const IMG_SELECT_BOX&& src);
        //IMG_SELECT_BOX& operator=(const IMG_SELECT_BOX& src);
        virtual ~IMG_SELECT_BOX();
        int get_select();
        void set_select(int select);
        void set_select_bttn_enable();
        void set_select_bttn_disable();
        bool get_select_bttn_sensitive();
        void set_image(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int dest_width, int dest_height, Gdk::InterpType interp_type);
        void set_image(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int dest_width, int dest_height);
        void set_image(Gdk::Pixbuf*& pixbuf, int dest_width, int dest_height, Gdk::InterpType interp_type);
        void clear_image();
        void save_image(const char* save_path, const char* type,int width, int height, Gdk::InterpType interp_type);
        void save_image(const char* save_path, const char* type);
        void save_image(const char* save_path, const char* type, int width, int height);
        void save_image(const char* save_path, int width, int height);
        void set_img_size_request(int width=-1, int height=-1);


    const static Gdk::InterpType DEFAULT_INTERP_TYPE; // =  Gdk::INTERP_BILENEAR
    const static char*  DEFAULT_IMG_EXTENSION;
        

    protected:
        Glib::RefPtr<Gdk::Pixbuf> scale_pixbuf(int dest_width, int dest_height, Gdk::InterpType interp_type);
        Glib::RefPtr<Gdk::Pixbuf> scale_pixbuf(int dest_width, int dest_height);
        void init(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf, int img_width, int img_height, Gdk::InterpType interp_type);


    int img_width = 0;
    int img_height = 0;
    bool select_buttons_sensitive = true;
    Gtk::Box WidgetContainer;
    Gtk::Image IMG;
    Glib::RefPtr<Gdk::Pixbuf> Orig_Pixbuf;
    Gtk::Box m_Separator;
    Gtk::Box RadioButtons;
    Gtk::RadioButton RB_Select; //radio button which present selection.
    Gtk::RadioButton RB_Discard; //radio button which present discard.

    public:
    
    inline Gtk::Widget& operator ()()
    {
        return WidgetContainer;
    }

};

#endif // RKG_IMG_SELECT_BOX_H