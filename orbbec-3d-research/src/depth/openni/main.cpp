#include <iostream>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>

using namespace std;


bool createDepthStream(openni::Device &device, openni::VideoStream& stream)
{
	openni::Status rc = stream.create(device, openni::SENSOR_DEPTH);
	if (rc != openni::STATUS_OK)
	{
		printf("Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
		return false;
	}

	return true;
}

bool readFrameFromStream(openni::VideoStream& stream, openni::VideoFrameRef& frame, int w, int h)
{
    // start specific stream
    stream.start();

    if (stream.isValid())
    {
        openni::VideoMode videoMode = stream.getVideoMode();
        videoMode.setResolution(w, h);
        stream.setVideoMode(videoMode);
    }

    openni::VideoStream* pStream = &stream;
    int changedIndex = -1;
    openni::Status rc = openni::OpenNI::waitForAnyStream(&pStream, 1, &changedIndex,10000);
    if (rc != openni::STATUS_OK)
    {   
        cout << rc << endl;
        printf("Wait failed\n");
        return false;
    }

    //Read the newest frame
    stream.readFrame(&frame);
    stream.stop();
    return true;
}

const int N = 150;

void depthToMat(const openni::VideoStream& stream, const openni::VideoFrameRef& frame, cv::Mat& mScaledDepth)
{
    // save as cv mat
    int iMaxDepth = stream.getMaxPixelValue();
    const cv::Mat mImageDepth(
                frame.getHeight(), frame.getWidth(),
                CV_16UC1, (void*)frame.getData() );
    mImageDepth.convertTo( mScaledDepth, CV_8U, 255.0 / iMaxDepth );
}

int main(int argc, char** argv)
{
    openni::Status rc;
    openni::Device device;
    openni::VideoStream depthStream;
    openni::VideoFrameRef depthFrame;

    cv::Mat depthImg;
    cv::Mat depthBgrImg;
    cv::Mat bgrImg;
    cv::Mat stichedImg;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()){ //This section prompt an error message if no video stream is found//
        cout << "No video stream detected" << endl;
        return-1;
    }

    cv::VideoWriter outVideo(
        "001.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cap.get(cv::CAP_PROP_FPS),
        cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH) * 2, cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    );

    rc = openni::OpenNI::initialize();
    rc = device.open(openni::ANY_DEVICE);

    // chech device and available stream types
    bool res = createDepthStream(device, depthStream);
	if (false == res)
	{
		openni::OpenNI::shutdown();
		return -1;
	}

    // start specific stream
    depthStream.start();

    if (depthStream.isValid())
    {
        openni::VideoMode videoMode = depthStream.getVideoMode();
        videoMode.setResolution(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        depthStream.setVideoMode(videoMode);
    }

    openni::VideoStream* pStream = &depthStream;
    int changedIndex = -1;
    rc = openni::OpenNI::waitForAnyStream(&pStream, 1, &changedIndex,10000);
    if (rc != openni::STATUS_OK)
    {   
        cout << rc << endl;
        printf("Wait failed\n");
        return -1;
    }

    for (int i =0; i < N; ++i){ //Taking an everlasting loop to show the video//
        cap >> bgrImg;
        depthStream.readFrame(&depthFrame);
        depthToMat(depthStream, depthFrame, depthImg);
        cv::flip(depthImg, depthImg, 1);

        cv::cvtColor(depthImg, depthBgrImg, cv::COLOR_GRAY2BGR );
        cv::hconcat(bgrImg, depthBgrImg, stichedImg);

        // Write the frame into the file 'outcpp.avi'
        outVideo.write(stichedImg);
    }

    cap.release();
    depthStream.stop();
    outVideo.release();
    return 0;
}