#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <stdio.h>
#include <thread>

using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;
CascadeClassifier fullbody_cascade;
CascadeClassifier upperbody_cascade;
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{face_cascade|data/haarcascades/haarcascade_frontalface_default.xml|Path to face cascade.}"
                             "{fullbody_cascade|data/haarcascades/haarcascade_fullbody.xml|Path to fullbody cascade.}"
                             "{upperbody_cascade|data/haarcascades/haarcascade_upperbody.xml|Path to upperbody cascade.}"
                             "{camera|1|Camera device number.}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
                  "You can use Haar or LBP features.\n\n" );
    parser.printMessage();
    String face_cascade_name = samples::findFile( parser.get<String>("face_cascade") );
    String fullbody_cascade_name = samples::findFile( parser.get<String>("fullbody_cascade") );
    String upperbody_cascade_name = samples::findFile( parser.get<String>("upperbody_cascade") );
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !fullbody_cascade.load( fullbody_cascade_name ))
    {
        cout << "--(!)Error loading fullbody cascade\n";
        return -1;
    }
    if( !upperbody_cascade.load( upperbody_cascade_name ))
    {
        cout << "--(!)Error loading upperbody cascade\n";
        return -1;
    }
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    // It is based on usb, if 0 then defaults
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    capture.open(deviceID + apiID);
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        waitKey(10);

    }
    return 0;
}

void personDetected( Mat frame )
{
    imwrite("testingImages/testerImage.jpg", frame );
    cout << "YES"<<endl;
}

void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    ///equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    ///std::vector<Rect> fullbodies;
    std::vector<Rect> upperbodies;

    face_cascade.detectMultiScale( frame_gray, faces );
    ///fullbody_cascade.detectMultiScale( frame_gray, fullbodies );
    upperbody_cascade.detectMultiScale( frame_gray, upperbodies );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 4 ); ///BLUE face
        Mat faceROI = frame_gray( faces[i] );
    }
    /*
    for ( size_t i =0; i < fullbodies.size(); i++)
    {
        Point center( fullbodies[i].x + fullbodies[i].width/2, fullbodies[i].y + fullbodies[i].height/2 );
        ellipse( frame, center, Size( fullbodies[i].width, fullbodies[i].height ), 0, 0, 360, Scalar( 0, 0, 255 ), 4 ); ///RED fullbody
        Mat faceROI = frame_gray( fullbodies[i] );
    }
    */
    for ( size_t i =0; i < upperbodies.size(); i++)
    {
        Point center( upperbodies[i].x + upperbodies[i].width/2, upperbodies[i].y + upperbodies[i].height/2 );
        ellipse( frame, center, Size( upperbodies[i].width, upperbodies[i].height ), 0, 0, 360, Scalar( 0, 255, 0 ), 4 ); ///GREEN upperbody
        Mat faceROI = frame_gray( upperbodies[i] );
    }
    if( faces.size() > 0 && upperbodies.size() > 2 )
    {
        thread thread_object(personDetected, frame_gray);
    }
    //-- Save image
    /*if(faces.size()>0 && fullbodies.size()>0 && upperbodies.size()>0)
    {
        cout << "\n\nOne Image";
        imwrite("testingImages\tester.jpg", frame_gray);
    }
    */
    //-- Display/s
    ///imshow( "Capture - gray ", frame_gray );
    imshow( "The normal output", frame);
}
