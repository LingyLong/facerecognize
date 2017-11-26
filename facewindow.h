#include <iostream>
#include <gtkmm.h>
#include <opencv2/opencv.hpp>
#include "extract_name.h"
#include "worker.h"
#include "extractfeature.h"
#include "select_video.h"

#include "util.h"

using namespace cv;
using namespace std;

class FaceWindow : public Gtk::Window
{
public:
	FaceWindow(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& refGlade);
	~FaceWindow();
protected:
	template <class T_Widget>
  bool create_dialog(const std::string & filename, const Glib::ustring & name, T_Widget*& widget);

//	int start_capture(std::string & device);

public:
	void on_start_button_clicked();
	void notify();
	void on_notification_from_worker_thread();

	void update_widgets();	

	void on_register();
	void on_recognize();
	void on_quit();
	
	std::string get_video_source();

	int read_all_features(std::string path, std::vector<std::pair<string, vector<float> > > & features);

private:
	std::string filename;
	Gtk::Allocation alloc;
	cv::Mat src;

	char keyboard=0;

	std::vector<std::pair<string, vector<float> > >  features;

  Glib::Dispatcher m_Dispatcher;
  Worker m_Worker;
  std::thread* m_WorkerThread;
	
	std::string source;
protected:
	Glib::RefPtr<Gtk::Builder> m_refBuilder;
	Gtk::Image* pImage1;
//	Gtk::Image* pImage2;

	Gtk::Button * pOpen;
	Gtk::Button * pRegister;
	Gtk::Button * pRecognize;
	Gtk::Button * pQuit;
};


