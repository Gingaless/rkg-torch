#ifndef RKG_IMG_TABLE_H
#define RKG_IMG_TABLE_H

#include "img_select_box.h"
#include <gtkmm/grid.h>

template <int rows, int cols, int w, int h> // w is width, h is height.
class IMG_TABLE
{
    public:
        /*template<int rows, int cols, int w, int h>*/ IMG_TABLE(Glib::RefPtr<Gdk::Pixbuf>** const (&img_refs));
        /*template<int rows, int cols, int w, int h>*/ IMG_TABLE();
        /*template<int rows, int cols>*/ void set_image_batch(Glib::RefPtr<Gdk::Pixbuf>** const (&img_refs));
        void set_image(const Glib::RefPtr<Gdk::Pixbuf>& img_ref, int row, int col);
        int get_rows();
        int get_cols();
        void get_image(int idx_rows, int idx_cols);
        void get_orig_image(int idx_rows, int idx_cols);
        void set_selects_enable();
        void set_selects_disable();
        virtual ~IMG_TABLE();

    protected:
        void init();
        void clear_image_boxes();


    Gtk::Grid img_grid;
    IMG_SELECT_BOX** img_select_boxes;
    int num_rows;
    int num_cols;
    int img_width;
    int img_height;

    public:

    inline Gtk::Widget& operator()()
    {
        return img_grid;
    }

    inline IMG_SELECT_BOX& operator()(int row, int col)
    {
        return img_select_boxes[row][col];
    }
};



template<int rows, int cols, int w, int h> IMG_TABLE<rows, cols, w, h>::IMG_TABLE(Glib::RefPtr<Gdk::Pixbuf>** const (&img_refs)) :
num_rows(rows),
num_cols(cols),
img_width(w),
img_height(h)
{
    init();
    set_image_batch(img_refs);
}

template<int rows, int cols, int w, int h> IMG_TABLE<rows, cols, w, h>::IMG_TABLE() :
num_rows(rows),
num_cols(cols),
img_width(w),
img_height(h)
{
    init();
}

template<int rows, int cols, int w, int h> void IMG_TABLE<rows, cols, w, h>::init()
{
    img_grid.set_row_homogeneous(true);
    img_grid.set_column_homogeneous(true);
    img_select_boxes = new IMG_SELECT_BOX*[num_rows];
    for (int i=0; i<num_rows; i++)
    {
        img_select_boxes[i] = new IMG_SELECT_BOX[num_cols];
        for (int j=0; j<num_cols; j++)
            img_grid.attach(img_select_boxes[i][j](), j, i);
    }
}

template<int rows, int cols, int w, int h> void IMG_TABLE<rows, cols, w, h>::set_image(const Glib::RefPtr<Gdk::Pixbuf>& img_ref, int row, int col)
{
    if (row < num_rows && col < num_rows)
    {
        img_select_boxes[row][col].set_image(img_ref, img_width, img_height);
    }
    else
    {
        throw std::out_of_range("image table index is out of range.");
    }
    
}

template<int rows, int cols, int w, int h> void IMG_TABLE<rows, cols, w, h>::set_image_batch(Glib::RefPtr<Gdk::Pixbuf>** const (&img_refs))
{
    for (int i=0; i<rows; i++)
        for (int j=0; j<cols; j++)
            set_image(img_refs[i][j], i, j);
}

template<int rows, int cols, int w, int h> void IMG_TABLE<rows, cols, w, h>::clear_image_boxes()
{
    for (int i=0; i<num_rows; i++)
        delete[] img_select_boxes[i];
    delete[] img_select_boxes;
}

template<int rows, int cols, int w, int h> IMG_TABLE<rows, cols, w, h>::~IMG_TABLE()
{
    clear_image_boxes();
}

 template<int rows, int cols, int w, int h> void IMG_TABLE<rows, cols, w, h>::set_selects_enable()
{
    for (int i=0; i<rows; i++)
        for (int j=0; j<cols; j++)
            img_select_boxes[i][j].set_select_bttn_enable();
}

template<int rows, int cols, int w, int h> void IMG_TABLE<rows, cols, w, h>::set_selects_disable()
{
    for (int i=0; i<rows; i++)
        for (int j=0; j<cols; j++)
            img_select_boxes[i][j].set_select_bttn_disable();
}



#endif // RKG_IMG_TABLE_H