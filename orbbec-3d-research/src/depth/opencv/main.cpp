#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <list>
#include <iostream>
#include <signal.h>

#define HAVE_THREADS 1


#if !defined(HAVE_THREADS)
int main()
{
    std::cout << "This sample is built without threading support. Sample code is disabled." << std::endl;
    return 0;
}
#else


#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;

const int W = 640;
const int H = 480;

// Stores frames along with their timestamps
struct Frame
{
    int64 timestamp;
    Mat frame;
};

//! [Open streams]
// Open depth stream
VideoCapture depthStream;
// Open color stream
VideoCapture colorStream;
VideoWriter outVideo;

std::thread depthReader;
std::thread colorReader;

std::condition_variable dataReady;
std::atomic<bool> isFinish;

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
   cout << "Caught signal " << signum << endl;

    isFinish = true;

    depthStream.release();
    colorStream.release();
    outVideo.release();

    dataReady.notify_one();
    depthReader.join();
    colorReader.join();

   // Terminate program
   exit(signum);
}

int main()
{
    // Register signal and signal handler
    signal(SIGINT, signal_callback_handler);

    Mat stichedImg;

    depthStream = VideoCapture(CAP_OPENNI2_ASTRA);
    colorStream = VideoCapture(0, CAP_V4L2);
    outVideo = VideoWriter(
        "001.avi",
        VideoWriter::fourcc('M', 'J', 'P', 'G'),
        colorStream.get(CAP_PROP_FPS),
        Size(W * 2, H)
    );
    
    //! [Open streams]

    // Check that stream has opened
    if (!colorStream.isOpened())
    {
        cerr << "ERROR: Unable to open color stream" << endl;
        return 1;
    }

    // Check that stream has opened
    if (!depthStream.isOpened())
    {
        cerr << "ERROR: Unable to open depth stream" << endl;
        return 1;
    }

    //! [Setup streams]
    // Set color and depth stream parameters
    colorStream.set(CAP_PROP_FRAME_WIDTH,  W);
    colorStream.set(CAP_PROP_FRAME_HEIGHT, H);
    depthStream.set(CAP_PROP_FRAME_WIDTH,  W);
    depthStream.set(CAP_PROP_FRAME_HEIGHT, H);
    depthStream.set(CAP_PROP_OPENNI2_MIRROR, 0);
    //! [Setup streams]

    // Print color stream parameters
    cout << "Color stream: "
        << colorStream.get(CAP_PROP_FRAME_WIDTH) << "x" << colorStream.get(CAP_PROP_FRAME_HEIGHT)
        << " @" << colorStream.get(CAP_PROP_FPS) << " fps" << endl;

    //! [Get properties]
    // Print depth stream parameters
    cout << "Depth stream: "
        << depthStream.get(CAP_PROP_FRAME_WIDTH) << "x" << depthStream.get(CAP_PROP_FRAME_HEIGHT)
        << " @" << depthStream.get(CAP_PROP_FPS) << " fps" << endl;
    //! [Get properties]

    //! [Read streams]
    // Create two lists to store frames
    std::list<Frame> depthFrames, colorFrames;
    const std::size_t maxFrames = 64;

    // Synchronization objects
    std::mutex mtx;

    isFinish = false;

    // Start depth reading thread
    depthReader = std::thread([&]
    {
        while (!isFinish)
        {
            // Grab and decode new frame
            if (depthStream.grab())
            {
                Frame f;
                f.timestamp = cv::getTickCount();
                depthStream.retrieve(f.frame, CAP_OPENNI_DEPTH_MAP);
                if (f.frame.empty())
                {
                    cerr << "ERROR: Failed to decode frame from depth stream" << endl;
                    break;
                }

                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (depthFrames.size() >= maxFrames)
                        depthFrames.pop_front();
                    depthFrames.push_back(f);
                }
                dataReady.notify_one();
            }
        }
    });

    // Start color reading thread
    colorReader = std::thread([&]
    {
        while (!isFinish)
        {
            // Grab and decode new frame
            if (colorStream.grab())
            {
                Frame f;
                f.timestamp = cv::getTickCount();
                colorStream.retrieve(f.frame);
                if (f.frame.empty())
                {
                    cerr << "ERROR: Failed to decode frame from color stream" << endl;
                    break;
                }

                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (colorFrames.size() >= maxFrames)
                        colorFrames.pop_front();
                    colorFrames.push_back(f);
                }
                dataReady.notify_one();
            }
        }
    });
    //! [Read streams]

    //! [Pair frames]
    // Pair depth and color frames
    while (!isFinish)
    {
        std::unique_lock<std::mutex> lk(mtx);
        while (!isFinish && (depthFrames.empty() || colorFrames.empty()))
            dataReady.wait(lk);

        while (!depthFrames.empty() && !colorFrames.empty())
        {
            if (!lk.owns_lock())
                lk.lock();

            // Get a frame from the list
            Frame depthFrame = depthFrames.front();
            int64 depthT = depthFrame.timestamp;

            // Get a frame from the list
            Frame colorFrame = colorFrames.front();
            int64 colorT = colorFrame.timestamp;

            // Half of frame period is a maximum time diff between frames
            const int64 maxTdiff = int64(1000000000 / (2 * colorStream.get(CAP_PROP_FPS)));
            if (depthT + maxTdiff < colorT)
            {
                depthFrames.pop_front();
                continue;
            }
            else if (colorT + maxTdiff < depthT)
            {
                colorFrames.pop_front();
                continue;
            }
            depthFrames.pop_front();
            colorFrames.pop_front();
            lk.unlock();

            //! [Show frames]
            // Show depth frame
            Mat d8, dColor;
            depthFrame.frame.convertTo(d8, CV_8U, 255.0 / 2500);
            applyColorMap(d8, dColor, COLORMAP_INFERNO);
            // imshow("Depth (colored)", dColor);

            // // Show color frame
            // imshow("Color", colorFrame.frame);
            //! [Show frames]
            cv::hconcat(colorFrame.frame, dColor, stichedImg);

            // Write the frame into the file 'outcpp.avi'
            outVideo.write(stichedImg);

        }
    }
    //! [Pair frames]

    return 0;
}

#endif