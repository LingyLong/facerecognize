#ifndef GTKMM_VIDEO_DIALOG_H
#define GTKMM_VIDEO_DIALOG_H
#include <iostream>
#include <gtkmm.h>
#include <opencv2/opencv.hpp>

class VideoDialog : public Gtk::Dialog
{
public:
  VideoDialog(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& refGlade);
  virtual ~VideoDialog();

protected:
  //Signal handlers:
	void on_button_ok();
  void on_button_cancel();


  Glib::RefPtr<Gtk::Builder> m_refbuilder;
  Gtk::Button* m_pButtonOK;
	Gtk::Button* m_pButtonCancel;
	
	Gtk::RadioButton *p1;
	Gtk::RadioButton *p2;

	Gtk::FileChooserButton *pfile;
public:
	bool get_result_value(int result, std::string & name);
public:	
	std::string name;
};

#endif //GTKMM_THRESHOLD_DIALOG_H
