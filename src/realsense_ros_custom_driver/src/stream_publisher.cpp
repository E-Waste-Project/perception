#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>
// This code requires several standard data-structures and algorithms:
#define _USE_MATH_DEFINES
#include <math.h>
#include <queue>
#include <unordered_set>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>
#include "ros/ros.h"
#include <unistd.h>
#include "realsense_ros_custom_driver/img_stream.h"


using namespace std;
using namespace std::chrono;

// Functions' Prototype definitions.
float get_depth_scale(rs2::device dev);
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
template <typename T>
void frame_to_arr(rs2::video_frame frame, std::vector<T> &arr, int n_channels = 1);

high_resolution_clock::time_point t_frame1;
high_resolution_clock::time_point t_frame2;

int main(int argc, char *argv[])try
{
  //ROS specifics:
  //----------------
  //Initializing ROS node with name stream_publisher.
  string node_name = "stream_publisher";
  ros::init(argc, argv, node_name);
  //Created a nodehandle object
  ros::NodeHandle node_obj;
  //Define a publisher for image streams.
  ros::Publisher img_stream_pub = node_obj.advertise<realsense_ros_custom_driver::img_stream>("/img_stream", 10);

  //camera variables:
  //-------------------

  // Streams' Parameters
  bool enable_depth, enable_color, enable_ir, align_depth;
  int depth_fps, color_fps, ir_fps;
  int depth_width, depth_height;
  int color_width, color_height;
  int ir_width, ir_height;

  // <!--depth stream-->
  node_obj.param<bool>("/" + node_name + "/enable_depth", enable_depth, false);
  node_obj.param<int>("/" + node_name + "/depth_fps", depth_fps, 30);
  node_obj.param<int>("/" + node_name + "/depth_width", depth_width, 1280);
  node_obj.param<int>("/" + node_name + "/depth_height", depth_height, 720);

  // <!--color stream-->
  node_obj.param<bool>("/" + node_name + "/enable_color", enable_color, true);
  node_obj.param<int>("/" + node_name + "/color_fps", color_fps, 30);
  node_obj.param<int>("/" + node_name + "/color_width", color_width, 1280);
  node_obj.param<int>("/" + node_name + "/color_height", color_height, 720);

  // <!--infrared stream-->
  node_obj.param<bool>("/" + node_name + "/enable_ir", enable_ir, true);
  node_obj.param<int>("/" + node_name + "/ir_fps", ir_fps, 30);
  node_obj.param<int>("/" + node_name + "/ir_width", ir_width, 1280);
  node_obj.param<int>("/" + node_name + "/ir_width", ir_width, 720);

  // <!--align_depth-->
  node_obj.param<bool>("/" + node_name + "/align_depth", align_depth, true);

  //Filters' Parameters
  bool enable_temporal_filter, enable_decimation_filter;
  bool enable_spatial_filter, enable_threshold_filter;
  float alpha, max_dist, min_dist;
  int delta, temporal_holes_fill, dec_value, spatial_holes_fill;

  // <!--temporal filter-->
  node_obj.param<bool>("/" + node_name + "/enable_temporal_filter", enable_temporal_filter, true);
  node_obj.param<float>("/" + node_name + "/alpha", alpha, 0.2);
  node_obj.param<int>("/" + node_name + "/delta", delta, 100);
  node_obj.param<int>("/" + node_name + "/temporal_holes_fill", temporal_holes_fill, 7); // <!--range is [0 => 8]-->

  // <!--decimation filter-->
  node_obj.param<bool>("/" + node_name + "/enable_decimation_filter", enable_decimation_filter, false);
  node_obj.param<int>("/" + node_name + "/dec_value", dec_value, 3);

  // <!--spatial filter-->
  node_obj.param<bool>("/" + node_name + "/enable_spatial_filter", enable_spatial_filter, false);
  node_obj.param<int>("/" + node_name + "/spatial_holes_fill", spatial_holes_fill, 5); // <!--range is [0 => 8]-->

  // <!--threshold filter-->
  node_obj.param<bool>("/" + node_name + "/enable_threshold_filter", enable_threshold_filter, false);
  node_obj.param<float>("/" + node_name + "/max_dist", max_dist, 1.4);
  node_obj.param<float>("/" + node_name + "/min_dist", min_dist, 0.1);

  // Filters
  rs2::decimation_filter dec;
  rs2::spatial_filter spat;
  rs2::hole_filling_filter holes;
  rs2::temporal_filter temp;
  rs2::threshold_filter thresh_filter;
  rs2::colorizer c;

  // Define transformations from and to Disparity domain
  rs2::disparity_transform depth2disparity;
  rs2::disparity_transform disparity2depth(false);

  // Processing options
  dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, dec_value);
  temp.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA,alpha);
  temp.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA,delta);
  temp.set_option(RS2_OPTION_HOLES_FILL,temporal_holes_fill);
  spat.set_option(RS2_OPTION_HOLES_FILL, spatial_holes_fill); // 5 = fill all the zero pixels
  thresh_filter.set_option(RS2_OPTION_MIN_DISTANCE, min_dist);
  thresh_filter.set_option(RS2_OPTION_MAX_DISTANCE, max_dist);

  //stream profile configuration and pipeline holders.
  rs2::pipeline pipe;
  rs2::config cfg;

  // Queues:
  // --------
  const unsigned int CAPACITY = 1; // allow max latency of 1 frame
  const unsigned int stream_CAPACITY = 1; // allow max latency of 1 frame
  // Original stream queue.
  rs2::frame_queue stream_frames(stream_CAPACITY);
  // After initial post-processing, frames will flow into this queue:
  rs2::frame_queue postprocessed_frames(CAPACITY);
  // Threading mutex to control queues access
  std::mutex mutex;

  // Define frame callback
  // The callback is executed on a sensor thread and can be called simultaneously from multiple sensors
  // Therefore any modification to common memory should be done under lock
  auto callback = [&](const rs2::frame& frame)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (rs2::frameset fs = frame.as<rs2::frameset>())
    {
      stream_frames.enqueue(fs);
    }
  };

  if(enable_depth){
    // Enable default depth stream.
    cfg.enable_stream(RS2_STREAM_DEPTH,depth_width, depth_height,RS2_FORMAT_Z16, depth_fps);
  }
  if(enable_color){
    // Enable RGB steam.
    cfg.enable_stream(RS2_STREAM_COLOR,color_width, color_height,RS2_FORMAT_RGB8, color_fps);
  }
  if(enable_ir){
    // Enable Infrared_1 stream (This is hardware aligned with the depth).
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, ir_width, ir_height, RS2_FORMAT_Y8, ir_fps);
  }

  auto profile = pipe.start(cfg,callback);

  //aligning of the color and depth streams
  rs2_stream align_to = RS2_STREAM_ANY;
  if(align_depth){
    align_to = find_stream_to_align(profile.get_streams());
  }
  rs2::align align(align_to);

  //geting sensor info
  auto sensor = profile.get_device().first<rs2::depth_sensor>();
  auto depth_scale = get_depth_scale(profile.get_device());

  // Alive boolean will signal the worker threads to finish-up
  std::atomic_bool alive{ true };

  //frame capturing and alignment thread:
  //-----------------------------------------
  std::thread video_processing_thread([&]() {
    // In order to generate new composite frames, we have to wrap the processing
    // code in a lambda
    rs2::processing_block frame_processor(
          [&](rs2::frameset data, // Input frameset (from the pipeline)
          rs2::frame_source& source) // Frame pool that can allocate new frames
    {
      t_frame1= high_resolution_clock::now();

      if(align_depth) data = data.apply_filter(align); //Here we align

      if(enable_temporal_filter || enable_spatial_filter){
        // To make sure far-away objects are filtered proportionally
        // we try to switch to disparity domain
        data = data.apply_filter(depth2disparity);
      }
      if(enable_temporal_filter){
        // Apply temporal filtering
        data = data.apply_filter(temp);
      }
      if(enable_spatial_filter){
        // Apply temporal filtering
        data = data.apply_filter(spat);
      }
      if(enable_temporal_filter || enable_spatial_filter){
        // If we are in disparity domain, switch back to depth
        data = data.apply_filter(disparity2depth);
      }

      // data = data.apply_filter(c);

      t_frame2 = high_resolution_clock::now();
      auto duration2 = duration_cast<milliseconds>(t_frame2-t_frame1).count();
      // cout<<"Alignment time="<<duration2<<endl;
      source.frame_ready(data);
    });

    // Indicate that we want the results of frame_processor
    // to be pushed into postprocessed_frames queue
    frame_processor >> postprocessed_frames;
    while (alive)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::lock_guard<std::mutex> lock(mutex);
      // Fetch frames from the pipeline and send them for processing
      rs2::frameset fs;
      // if (pipe.poll_for_frames(&fs)) frame_processor.invoke(fs);
      if(stream_frames.poll_for_frame(&fs))
      {
        // cout<<"thread__in"<<endl;
        frame_processor.invoke(fs);
      }
    }
  });
  // Define 1D arrays that will be published (An easier & faster way to publish images)
  std::vector<double_t> depth_arr;
  std::vector<uint8_t> bgr_arr, ir_arr;
  rs2_intrinsics intr;

  while (ros::ok())
  {
    // Make sure to clear arrays each loob iteration.
    depth_arr.clear();
    bgr_arr.clear();
    ir_arr.clear();

    // A frameset holder used to store the new processed frames from each enabled stream.
    static rs2::frameset current_frameset;

    // if postprocessed_frames queue has a new frameset enter the if clause and ,
    // store it in current_frameset.
    if(postprocessed_frames.poll_for_frame(&current_frameset))
    {
      // Define the messege holder to be published.
      realsense_ros_custom_driver::img_stream img_stream_msg;

      // Get each stream frame from current_frameset.
      if(enable_color){
        auto color_frame = current_frameset.get_color_frame();
        // Store the images' data in the 1D arrays (The data types should be as in the msg definition).
        frame_to_arr<uint8_t>(color_frame, bgr_arr, 3); // uint8 in msg definition.
        // store the data to be puuclished in the img_stream_msg messege holder.
        img_stream_msg.bgr = bgr_arr;
        img_stream_msg.color_width = color_width;
        img_stream_msg.color_height = color_height;
      }

      if(enable_depth){
        auto depth_frame = current_frameset.get_depth_frame();
        // Store the images' data in the 1D arrays (The data types should be as in the msg definition).
        frame_to_arr<double_t>(depth_frame, depth_arr); // float64 in msg definition.
        // Get the intrinsics of the camera to be published (Used to get 3d distance from pixels).
        intr = depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
        // store the data to be puuclished in the img_stream_msg messege holder.
        img_stream_msg.time_stamp = depth_frame.get_timestamp();
        img_stream_msg.depth = depth_arr;
        img_stream_msg.depth_scale = depth_scale * depth_frame.get_units();
        img_stream_msg.depth_width = intr.width;
        img_stream_msg.depth_height = intr.height;
        img_stream_msg.ppx = intr.ppx;
        img_stream_msg.ppy = intr.ppy;
        img_stream_msg.fx = intr.fx;
        img_stream_msg.fy = intr.fy;
        img_stream_msg.model = intr.model;
        img_stream_msg.coeffs.assign(intr.coeffs, intr.coeffs + 5);
      }

      if(enable_ir){
        auto ir_frame = current_frameset.get_infrared_frame();
        // Store the images' data in the 1D arrays (The data types should be as in the msg definition).
        frame_to_arr<uint8_t>(ir_frame, ir_arr); // uint8 in msg definition.
        // store the data to be puuclished in the img_stream_msg messege holder.
        img_stream_msg.ir = ir_arr;
        img_stream_msg.ir_width = ir_width;
        img_stream_msg.ir_height = ir_height;
      }   

      // Publish the img_stream_msg.
      img_stream_pub.publish(img_stream_msg);
    }
  }
  pipe.stop();
  alive = false;
  video_processing_thread.join();
  return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor is a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}


rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}


template <typename T>
void frame_to_arr(rs2::video_frame frame, std::vector<T> &arr, int n_channels)
{
  int pixels_per_channel = frame.get_height() + frame.get_width();
  arr.assign((T *)frame.get_data(), (T *)frame.get_data() + pixels_per_channel * n_channels);
}
