#include <iostream>
#include <gtkmm.h>
#include "facewindow.h"

int main(int argc, char *argv[])
{
  Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "org.facewindow");
  Glib::RefPtr<Gtk::Builder> m_refBuilder;
	try{
    m_refBuilder=Gtk::Builder::create_from_file("winface1.xml");
  }
  catch(const Glib::Error& ex){
    std::cout << "Building menus and toolbar failed: " << ex.what();
  }
	
	FaceWindow *pwin;
	m_refBuilder->get_widget_derived("winface1",pwin);
	
  return app->run(*pwin);
}

