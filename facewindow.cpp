#include <fstream>
#include "facewindow.h"


FaceWindow::FaceWindow(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& refGlade)
:Gtk::Window(cobject),
 m_refBuilder(refGlade),
 pRegister(nullptr), pOpen(nullptr),
 pRecognize(nullptr), pQuit(nullptr),
 pImage1(nullptr),
  m_Dispatcher(),
  m_Worker(),
  m_WorkerThread(nullptr)
{
	set_default_size(640,480);

	m_refBuilder->get_widget("btnOpen", pOpen);
	if(pOpen)
		pOpen->signal_clicked().connect(sigc::mem_fun(*this, &FaceWindow::on_start_button_clicked));

	m_refBuilder->get_widget("btnRegister", pRegister);
	if(pRegister)
		pRegister->signal_clicked().connect(sigc::mem_fun(*this, &FaceWindow::on_register));
	
	m_refBuilder->get_widget("btnRecognize", pRecognize);
	if(pRecognize)
		pRecognize->signal_clicked().connect(sigc::mem_fun(*this, &FaceWindow::on_recognize));

	m_refBuilder->get_widget("btnQuit", pQuit);
	if(pQuit)
		pQuit->signal_clicked().connect(sigc::mem_fun(*this, &FaceWindow::on_quit));

	m_refBuilder->get_widget("image1", pImage1);
//	m_refBuilder->get_widget("image2", pImage2);

  m_Dispatcher.connect(sigc::mem_fun(*this, &FaceWindow::on_notification_from_worker_thread));


	caffe_predefine();

	string path="./features/";
	read_all_features(path, features);
	
	show_all_children();
}

FaceWindow::~FaceWindow()
{

}
template <class T_Widget>
bool FaceWindow::create_dialog(const std::string & filename, const Glib::ustring & name, T_Widget*& widget)
{
  auto refbuilder=Gtk::Builder::create();
  try{
    refbuilder->add_from_file(filename);
  }
  catch(const Glib::FileError &ex){
    std::cerr<<"FileError: "<<ex.what()<<std::endl;
    return false;
  }
  catch(const Glib::MarkupError &ex){
    std::cerr<<"MarkupError: "<<ex.what()<<std::endl;
    return false;
  }
  catch(const Gtk::BuilderError & ex){
    std::cerr<<"BuilderError: "<<ex.what()<<std::endl;
    return false;
  }

  refbuilder->get_widget_derived(name, widget);

  return true;
}

std::string FaceWindow::get_video_source()
{
	return source;
}

void FaceWindow::on_start_button_clicked()
{
    VideoDialog *pdialog=0;
    create_dialog("xml/video_source.xml", "dialog1", pdialog);
    if(pdialog){
      pdialog->set_transient_for(*this);
      int result=pdialog->run();
      pdialog->get_result_value(result,source);
    }

	
  if (m_WorkerThread)
  {
    std::cout << "Can't start a worker thread while another one is running." << std::endl;
  }
  else
  {
    // Start a new worker thread.
    m_WorkerThread = new std::thread(
      [this]
      {
        m_Worker.do_work(this);
      });
  }
}


void FaceWindow::on_register()
{
	std::string username;

	if(!src.data)
		return ;

		
	cvtColor(src, src, CV_RGB2BGR);
		
	std::vector<cv::Mat> faces=detectFaces(src);
	for(int i=0; i<faces.size();i++){
		cv::resize(faces[i], faces[i], Size(224,224));
  	vector<float> face_vec=extractfeature(faces[i]);


		ExtractDialog *pdialog=0;
		create_dialog("dlgname.xml", "dlgname", pdialog);
		if(pdialog){
 			pdialog->set_transient_for(*this);
			cvtColor(faces[i], faces[i], CV_BGR2RGB);
			pdialog->set_image(faces[i]);	
			int result=pdialog->run();
			pdialog->get_result_value(result,username);
		}
		if(username.length()>0){	
 			std::string name_txt="features/"+username+".txt";
			std::pair<string, vector<float> > temp=make_pair(username, face_vec);
			features.push_back(temp);

			ofstream myfile(name_txt, ios::app);
			int j=0;
			while(j<face_vec.size())
				myfile << face_vec[j++]<<std::endl;
		}
	}
		
}


int FaceWindow::read_all_features(std::string path, std::vector<std::pair<string, vector<float> > > & features)
{
	std::vector<std::string> filename=read_features(path);
	
	if(filename.size()==0)
		return -1;

	std::vector<float> feature_one;
	std::string username;

	for(int k=0; k<filename.size(); k++){
			feature_one.clear();
			std::string name_txt=path+filename[k];
			ifstream myfile(name_txt, ios::in);
			int j=0;
			while(j<2622){
				float a;
				myfile >>a;
				feature_one.push_back(a);
				j++;
			}
			username=filename[k].substr(0,filename[k].length()-4);
			pair<string, vector<float> > temp=make_pair(username, feature_one);		
			features.push_back(temp);
	} 

	return 0;

}

void FaceWindow::on_recognize()
{
	if(!src.data)
		return ;

	if(features.size()==0){
		std::cout<<"no features data at all\n";
		return ;
	}
	std::vector<pair<cv::Mat, cv::Rect> > faces_all;
	std::string username;

	cvtColor(src, src, CV_RGB2BGR);	
	
	std::vector<float> face_vec;
	float coeff=0;
	float cosdist=0;
	cv::Mat face_one;
	detectFaces(src, faces_all);

	for(int i=0; i<faces_all.size();i++){
		face_vec.clear();
		coeff=0;
		face_one=faces_all[i].first;

		cv::resize(face_one, face_one, Size(224,224));
  	face_vec=extractfeature(face_one);
		for(int k=0;k<features.size();k++){ 		
			std::vector<float> feature_one=features[k].second;
			float t=coefficient(face_vec, feature_one);
			float cost=cos_distance(face_vec, feature_one);
			if(coeff<t){
				coeff=t;
				username=features[k].first;		
			}
			if(cosdist<cost){
				cosdist=cost;
				username=features[k].first;
			}
		}
		std::cout<<"euler dist:\nusername: "<<username<<", coeff: "<<coeff<<std::endl;
		std::cout<<"cos dist:\nusername: "<<username<<", cosdist: "<<cosdist<<std::endl;
//		cv::Rect face_rect=faces_all[i].second;
//		cv::putText(face_one, username, cv::Point(face_rect.x, face_rect.y),cv::FONT_HERSHEY_SIMPLEX ,1, Scalar(255,0,255));	

		ExtractDialog *pdialog=0;
		create_dialog("dlgname.xml", "dlgname", pdialog);
		if(pdialog){
 			pdialog->set_transient_for(*this);
			cvtColor(face_one, face_one, CV_BGR2RGB);
			pdialog->set_image(face_one);	
			pdialog->set_name(username);
			pdialog->set_timer(1);
			int result=pdialog->run();
		}
		
	}
	
	return ;

}
void FaceWindow::notify()
{
  m_Dispatcher.emit();
}
void FaceWindow::on_notification_from_worker_thread()
{
  if (m_WorkerThread && m_Worker.has_stopped())
  {
    // Work is done.
    if (m_WorkerThread->joinable())
      m_WorkerThread->join();
    delete m_WorkerThread;
    m_WorkerThread = nullptr;
  }
  update_widgets();
}

void FaceWindow::update_widgets()
{
	m_Worker.get_data(src);
	if(!src.data)
		return ;
	cvtColor(src, src, CV_BGR2RGB);

	Glib::RefPtr<Gdk::Pixbuf> pixbuf =Gdk::Pixbuf::create_from_data((guint8*)src.data,Gdk::COLORSPACE_RGB,false,8,src.cols,src.rows,src.step);
	pImage1->set(pixbuf);
	pImage1->queue_draw();
		

}

void FaceWindow::on_quit()
{
  if (m_WorkerThread)
  {
    // Order the worker thread to stop and wait for it to stop.
    m_Worker.stop_work();
    if (m_WorkerThread->joinable())
      m_WorkerThread->join();
  }
	close();
}
	
