#ifndef GTKMM_EXTRACT_DIALOG_H
#define GTKMM_EXTRACT_DIALOG_H
#include <iostream>
#include <gtkmm.h>
#include <opencv2/opencv.hpp>

class ExtractDialog : public Gtk::Dialog
{
public:
  ExtractDialog(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& refGlade);
  virtual ~ExtractDialog();

protected:
  //Signal handlers:
	void on_button_ok();
  void on_button_cancel();

	bool on_timeout(int timer_number);

  Glib::RefPtr<Gtk::Builder> m_refbuilder;
  Gtk::Button* m_pButtonOK;
	Gtk::Button* m_pButtonCancel;
	
	Gtk::Entry * pName;
	Gtk::Image * pImage;
public:
	bool get_result_value(int result, std::string & name);
	void set_image(cv::Mat &image);
	void set_name(std::string & username);
	void set_timer(int timer_number);
public:	
	std::string name;
	int timeout_value;
};

#endif //GTKMM_THRESHOLD_DIALOG_H
