#include "extract_name.h"

ExtractDialog::ExtractDialog(BaseObjectType* cobject, const Glib::RefPtr<Gtk::Builder>& refGlade)
: Gtk::Dialog(cobject),
  m_refbuilder(refGlade),
  m_pButtonOK(nullptr),
	m_pButtonCancel(nullptr),
	pName(nullptr)
{

	set_type_hint(Gdk::WINDOW_TYPE_HINT_SPLASHSCREEN);
  //Get the Glade-instantiated Button, and connect a signal handler:
  m_refbuilder->get_widget("btnOK", m_pButtonOK);
  if(m_pButtonOK)
  {
    m_pButtonOK->signal_clicked().connect( sigc::mem_fun(*this, &ExtractDialog::on_button_ok) );
  }

  m_refbuilder->get_widget("btnCancel", m_pButtonCancel);
  if(m_pButtonCancel)
  {
    m_pButtonCancel->signal_clicked().connect( sigc::mem_fun(*this, &ExtractDialog::on_button_cancel) );
  }
	
  m_refbuilder->get_widget("entry_name", pName);
  m_refbuilder->get_widget("image1", pImage);



}

// The first two parameters are mandatory in a constructor that will be called
// from Gtk::Builder::get_widget_derived().
// Additional parameters, if any, correspond to additional arguments in the call
// to Gtk::Builder::get_widget_derived().

ExtractDialog::~ExtractDialog()
{
}

void ExtractDialog::on_button_cancel()
{
	response(Gtk::RESPONSE_CANCEL);	
	hide();
}
void ExtractDialog::on_button_ok()
{
	name=pName->get_text();

	response(Gtk::RESPONSE_OK);
	hide();
}
bool ExtractDialog::get_result_value(int result, std::string & name)
{
    switch(result){
    case Gtk::RESPONSE_OK:
			name=pName->get_text();
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
void ExtractDialog::set_image(cv::Mat & image)
{
  Glib::RefPtr<Gdk::Pixbuf> pixbuf =Gdk::Pixbuf::create_from_data((guint8*)image.data, Gdk::COLORSPACE_RGB, false, 8, image.cols, image.rows, image.step);
  auto pixbufscaled=pixbuf->scale_simple(200,200,Gdk::INTERP_BILINEAR);
  pImage->set(pixbufscaled);
  pImage->queue_draw();

}
void ExtractDialog::set_name(std::string & username)
{
	if(pName)
		pName->set_text(username);
}

void ExtractDialog::set_timer(int timer_number)
{
	timeout_value=3000;

	sigc::slot<bool> my_slot=sigc::bind(sigc::mem_fun(*this, &ExtractDialog::on_timeout), timer_number);
	sigc::connection conn=Glib::signal_timeout().connect(my_slot, timeout_value);
	
}

bool ExtractDialog::on_timeout(int timer_number)
{
//	sleep(2);
//	signal_response(Gtk::RESPONSE_OK);
	on_button_ok();
}
