#include "worker.h"
#include "facewindow.h"
#include <sstream>
#include <chrono>

Worker::Worker() :
  m_Mutex(),
  m_shall_stop(false),
  m_has_stopped(false)
{
}

// Accesses to these data are synchronized by a mutex.
// Some microseconds can be saved by getting all data at once, instead of having
// separate get_fraction_done() and get_message() methods.
void Worker::get_data(cv::Mat & src) const
{
  std::lock_guard<std::mutex> lock(m_Mutex);

  if (frame.data)
    src=frame.clone();
}

void Worker::stop_work()
{
  std::lock_guard<std::mutex> lock(m_Mutex);
  m_shall_stop = true;
}

bool Worker::has_stopped() const
{
  std::lock_guard<std::mutex> lock(m_Mutex);
  return m_has_stopped;
}

void Worker::do_work(FaceWindow* caller)
{
	VideoCapture capture(caller->get_video_source());
//	VideoCapture capture("/dev/video0");

  {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_has_stopped = false;
  } // The mutex is unlocked here by lock's destructor.

  // Simulate a long calculation.
  while(true) // do until break
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    {
      std::lock_guard<std::mutex> lock(m_Mutex);
			
			capture >> frame;		
	
    }

    caller->notify();
  }

  {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_shall_stop = false;
    m_has_stopped = true;
  }

  caller->notify();
}
