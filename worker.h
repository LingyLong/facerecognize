#ifndef GTKMM_EXAMPLEWORKER_H
#define GTKMM_EXAMPLEWORKER_H

#include <gtkmm.h>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

class FaceWindow;

class Worker
{
public:
  Worker();

  // Thread function.
  void do_work(FaceWindow* caller);

  void get_data(cv::Mat & src) const;
  void stop_work();
  bool has_stopped() const;

private:
  // Synchronizes access to member data.
  mutable std::mutex m_Mutex;

  // Data used by both GUI thread and worker thread.
  bool m_shall_stop;
  bool m_has_stopped;
  
  cv::Mat frame;
};

#endif // GTKMM_EXAMPLEWORKER_H
