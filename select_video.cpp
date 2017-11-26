#include "select_video.h"

VideoDialog::VideoDialog(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& refGlade)
: Gtk::Dialog(cobject),
  m_refbuilder(refGlade),
  m_pButtonOK(nullptr),
	m_pButtonCancel(nullptr),
	p1(nullptr),p2(nullptr), pfile(nullptr)
{
  //Get the Glade-instantiated Button, and connect a signal handler:
  m_refbuilder->get_widget("btnOK", m_pButtonOK);
  if(m_pButtonOK)
  {
    m_pButtonOK->signal_clicked().connect( sigc::mem_fun(*this, &VideoDialog::on_button_ok) );
  }

  m_refbuilder->get_widget("btnCancel", m_pButtonCancel);
  if(m_pButtonCancel)
  {
    m_pButtonCancel->signal_clicked().connect( sigc::mem_fun(*this, &VideoDialog::on_button_cancel) );
  }
	
  m_refbuilder->get_widget("webcam", p1);
	
  m_refbuilder->get_widget("videofile", p2);

	p2->join_group(*p1);
	p1->set_active();

	m_refbuilder->get_widget("filechooserbutton1", pfile);
	
}

// The first two parameters are mandatory in a constructor that will be called
// from Gtk::Builder::get_widget_derived().
// Additional parameters, if any, correspond to additional arguments in the call
// to Gtk::Builder::get_widget_derived().

VideoDialog::~VideoDialog()
{
}

void VideoDialog::on_button_cancel()
{
	response(Gtk::RESPONSE_CANCEL);	
	hide();
}
void VideoDialog::on_button_ok()
{
	if(p1->get_active())
		name="/dev/video0";
	if(p2->get_active())
		name=pfile->get_filename();

	response(Gtk::RESPONSE_OK);
	hide();
}
bool VideoDialog::get_result_value(int result, std::string & name)
{
    switch(result){
    case Gtk::RESPONSE_OK:
			name=this->name;
      break;
    case Gtk::RESPONSE_CANCEL:
      std::cout<<"cancel clicked"<<std::endl;
      return false;
      break;
    default:
      std::cout<<"unknow "<<std::endl;
      return false;
      break;
    }
		return true;

}

