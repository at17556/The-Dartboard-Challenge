/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Global variables */
// String cascade_name = "frontalface.xml";
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

int dartNo = 0;

void drawManual(Mat frame) {
	if( dartNo == 0 ) {
		cv::rectangle(frame, cv::Rect(423, 3, 197, 214), cv::Scalar(0, 0, 255), 2);
	}
	if( dartNo == 1 ) { cv::rectangle(frame, cv::Rect(163, 102, 254, 252), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 2 ) { cv::rectangle(frame, cv::Rect(88, 82, 116, 115), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 3 ) { cv::rectangle(frame, cv::Rect(310, 138, 91, 95), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 4 ) { cv::rectangle(frame, cv::Rect(153, 68, 270, 259), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 5 ) { cv::rectangle(frame, cv::Rect(416, 128, 134, 139), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 6 ) { cv::rectangle(frame, cv::Rect(203, 110, 81, 80), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 7 ) { cv::rectangle(frame, cv::Rect(233, 149, 180, 187), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 8 ) {
		cv::rectangle(frame, cv::Rect(827, 203, 150, 152), cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, cv::Rect(63, 237, 69, 112), cv::Scalar(0, 0, 255), 2);
	}
	if( dartNo == 9 ) { cv::rectangle(frame, cv::Rect(165, 12, 304, 307), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 10 ) {
		cv::rectangle(frame, cv::Rect(75, 91, 123, 136), cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, cv::Rect(578, 120, 68, 102), cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, cv::Rect(915, 141, 38, 81), cv::Scalar(0, 0, 255), 2);
	}
	if( dartNo == 11 ) { cv::rectangle(frame, cv::Rect(167, 94, 73, 102), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 12 ) { cv::rectangle(frame, cv::Rect(150, 57, 75, 173), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 13 ) { cv::rectangle(frame, cv::Rect(253, 102, 166, 173), cv::Scalar(0, 0, 255), 2); }
	if( dartNo == 14 ) {
		cv::rectangle(frame, cv::Rect(100, 80, 167, 169), cv::Scalar(0, 0, 255), 2);
		cv::rectangle(frame, cv::Rect(967, 78, 163, 163), cv::Scalar(0, 0, 255), 2);
	}
	if( dartNo == 15 ) { cv::rectangle(frame, cv::Rect(129, 35, 174, 176), cv::Scalar(0, 0, 255), 2); }
}

float calculatef1(int tp, int fp, int fn) {
	float f1 = ( float ) (2 * tp) / ( float) ((2 * tp) + fp + fn);

	return f1;
}

float returnIoU(int rectA_x, int rectA_y, int rectA_x2, int rectA_y2, int rectB_x, int rectB_y, int rectB_x2, int rectB_y2, int i, int d) {

	if(rectA_x > rectB_x2 || rectB_x > rectA_x2) {
		return 0;
	}

	if(rectA_y > rectB_y2 || rectB_y > rectA_y2) {
		return 0;
	}

	int xA = max(rectA_x, rectB_x);
	int yA = max(rectA_y, rectB_y);
	int xB = min(rectA_x2, rectB_x2);
	int yB = min(rectA_y2, rectB_y2);

	float intersection = ((xB - xA) + 1) * ((yB - yA) + 1);

	float areaA = (rectA_x2 - rectA_x + 1) * (rectA_y2 - rectA_y + 1);
	float areaB = (rectB_x2 - rectB_x + 1) * (rectB_y2 - rectB_y + 1);

	float unionn = areaA + areaB - intersection;

	float IoU = intersection / unionn;

	printf("Machine face: %d, Actual face %d, Intersection: %f, Union: %f, IoU: %f, \n", i, d, intersection, unionn, IoU);

	return IoU;
}

void calcMagnitudeOfGradient(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &magnitudeOutput) {
  assert(dx.size() == dy.size());
  magnitudeOutput.create(dx.size());

  for (int i = 0; i < dx.rows; i++) {
    for (int j = 0; j < dx.cols; j++) {
      float x = dx.at<float>(i, j);
      float y = dy.at<float>(i, j);

      magnitudeOutput.at<float>(i, j) = sqrt(x*x + y*y);
    }
  }
}

void calcDirectionOfGradient(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &gradientOutput) {
	assert(dx.size() == dy.size());
	gradientOutput.create(dx.size());

	for (int i = 0; i < dx.rows; i++) {
		for (int j = 0; j < dx.cols; j++) {
			float x = dx(i, j);
			float y = dy(i, j);

			gradientOutput.at<float>(i, j) = atan2(y, x);
		}
	}
}


void applyKernel(const cv::Mat_<uchar> &input, cv::Mat_<float> &kernel, cv::Mat_<float> &output ) {
	// intialise the output using the input
	output.create(input.size());

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat_<uchar> paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			float sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					float kernalval = kernel.at<float>( kernelx, kernely );

					// do the multiplication
					sum += ( float ) imageval * kernalval;
				}
			}
			// printf("%f", sum);
			// set the output value as the sum of the convolution
			output.at<float>( i, j ) = sum;
		}

	}
}

void normalise(cv::Mat &image, cv::Mat &output){
  float min = 999.0;
  float max = -999.0;

  for ( int x = 0; x < image.rows; x++ ) {
 	 for( int y = 0; y < image.cols; y++ ) {
     // image.at<float>(x, y) *= 10;
 		 float current = image.at<float>(x, y);
     // printf("%f\n", current);
 		 if (current < min) min = current;
 		 if (current > max) max = current;


 	 }
 }

  // printf("\n %i, %i \n", min, max);

  for ( int x = 0; x < image.rows; x++ ) {
  	for( int y = 0; y < image.cols; y++ ) {

  		 float current = image.at<float>(x, y);
       // output.at<uchar>(x, y) = (image.at<float>(x, y) - min) / (max/255);
 		   output.at<uchar>(x, y) = ((current - min) * ((255 - 0)/(max - min)));
       // printf("%f\n", output.at<uchar>(x, y));
  	}
  }
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame ) {

	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

	// Draw green rectangle
	drawManual( frame );

	// HOUGH TRANSFORM
	cv::Mat_<uchar> gray_image;
	cvtColor( frame, gray_image, CV_BGR2GRAY );

	// Apply sobel kernel for X direction
	cv::Mat_<float> sobelX;
	cv::Mat_<float> kernelX(3, 3);
	kernelX << -1, 0, 1, -2, 0, 2, -1, 0, 1;
	applyKernel(gray_image, kernelX, sobelX);

	// Apply sobel kernel for Y direction
	cv::Mat_<float> sobelY;
	cv::Mat_<float> kernelY(3, 3);
	kernelY << -1, -2, -1, 0, 0, 0, 1, 2, 1;
	applyKernel(gray_image, kernelY, sobelY);

	// Find magnitude
	cv::Mat_<float> magOfGradient;
	calcMagnitudeOfGradient(sobelX, sobelY, magOfGradient);

	// Find Direction
	cv::Mat_<float> dirOfGradient;
	calcDirectionOfGradient(sobelX, sobelY, dirOfGradient);

	// Hough algo
	int threshold = 48;
	int minRadius = 25;
	int maxRadius = 128;

	float thetaThreshold = 0.120;
	float thetaIncr = (2*M_PI) / 360;

	cv::Mat Hough;
	int parameters[] = {magOfGradient.rows + maxRadius, magOfGradient.cols + maxRadius, maxRadius};
	Hough.create(3, parameters, ( float ) magOfGradient.type());

	cv::Mat drawHough;
	drawHough.create(magOfGradient.rows, magOfGradient.cols, ( float ) magOfGradient.type());

	for (int i = 0; i < magOfGradient.rows; i++) {
		for (int j = 0; j < magOfGradient.cols; j++) {

			if (magOfGradient.at<float>(i,j) > threshold) {

				for (int r = minRadius; r < maxRadius; r++) {

					float direction = dirOfGradient.at<float>(i, j);

					for (float theta = direction - thetaThreshold; theta < direction + thetaThreshold; theta += thetaIncr) {
						for (int m = -1; m <= 1; m = m + 2) {
							for (int n = -1; n <= 1; n = n + 2) {
								int x0 = ( int ) j + (m*r) * cosf(theta);
								int y0 = ( int ) i + (n*r) * sinf(theta);
								if (x0 < magOfGradient.cols && x0 > 0) {
									if (y0 < magOfGradient.rows && y0 > 0) {
										Hough.at<float>(y0 , x0, r)++;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// Draw Circles
	int minVotes = 150;
	float min = 9999.0;
	float max = 0.0;
	int centreX;
	int centreY;
	int noOfVotes;

	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			for (int r = minRadius; r < maxRadius; r++) {
				noOfVotes = Hough.at<float>(y, x, r);
				if (noOfVotes > minVotes) {
					circle(frame, Point(x, y), r, cvScalar(255,0,0), 1);
					if(r > max) {
						max = r;
						centreX = x;
						centreY = y;
					}
				}
			}
		}
	}

	// printf("CENTREX: %d", centreX);
	int topLeftX = centreX - max;
	int topLeftY = centreY - max;


	float maximum = 0.0;
	int detected = 0;
	int rectNumber;
	float thresholdIoU = 0.45;

	// rectangle(frame, Point(topLeftX, topLeftY), Point(topLeftX + 2 * max, topLeftY + 2 * max), Scalar( 0, 0, 255 ), 2);

  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ ) {

		// Print machine detected boxes
		// rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);

		// Producing the IoU from 2 given rectangles
		int d = 0;
		int xA = topLeftX;
		int yA = topLeftY;
		int x2A = topLeftX + 2 * max;
		int y2A = topLeftY + 2 * max;
		int xB = faces[i].x;
		int yB = faces[i].y;
		int x2B = faces[i].x + faces[i].width;
		int y2B = faces[i].y + faces[i].height;

		float IoU = returnIoU(xA, yA, x2A, y2A, xB, yB, x2B, y2B, i, d);

		// if (IoU > threshold ) {
		// 	detected++;
		// }

		if( IoU > maximum ) {
			maximum = IoU;
			rectNumber = i;
		}
	}

	if(maximum > thresholdIoU) {
		rectangle(frame, Point(faces[rectNumber].x, faces[rectNumber].y), Point(faces[rectNumber].x + faces[rectNumber].width, faces[rectNumber].y + faces[rectNumber].height), Scalar( 0, 255, 0 ), 2);
	}


	int p = 1;
	int tp = detected;
	int fp = faces.size() - tp;
	int fn = p - tp;
	float f1_score = calculatef1(tp, fp, fn);

	printf("Actual faces (p): %d\n", p);
	printf("Detected correct faces (tp): %d\n", tp);
	printf("Detected wrong faces (fp): %d\n", fp);
	printf("Undetected faces (fn): %d\n", fn);
	printf("f1 score: %f\n", f1_score);

	for(int x = 0; x < magOfGradient.rows; x++){
	 for(int y = 0; y < magOfGradient.cols; y++){
		 for(int r = minRadius; r < maxRadius; r++){
			 drawHough.at<float>(x, y) += Hough.at<float>(x, y, r);
		 }
	 }
	}

	// cv::Mat output;
	// output.create(magOfGradient.rows, magOfGradient.cols, ( uchar ) gray_image.type());
	//
	// normalise(drawHough, output);
	// imwrite( "HoughTransform.jpg", output);

	// normalise(magOfGradient, output);
	// imwrite( "magOfGradient.jpg", output);
}

/** @function main */
int main( int argc, const char** argv ) {

  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  String imageName = argv[1];

	if(imageName == "dart0.jpg") { dartNo = 0; }
	if(imageName == "dart1.jpg") { dartNo = 1; }
	if(imageName == "dart2.jpg") { dartNo = 2; }
	if(imageName == "dart3.jpg") { dartNo = 3; }
	if(imageName == "dart4.jpg") { dartNo = 4; }
	if(imageName == "dart5.jpg") { dartNo = 5; }
	if(imageName == "dart6.jpg") { dartNo = 6; }
	if(imageName == "dart7.jpg") { dartNo = 7; }
	if(imageName == "dart8.jpg") { dartNo = 8; }
	if(imageName == "dart9.jpg") { dartNo = 9; }
	if(imageName == "dart10.jpg") { dartNo = 10; }
	if(imageName == "dart11.jpg") { dartNo = 11; }
	if(imageName == "dart12.jpg") { dartNo = 12; }
	if(imageName == "dart13.jpg") { dartNo = 13; }
	if(imageName == "dart14.jpg") { dartNo = 14; }
	if(imageName == "dart15.jpg") { dartNo = 15; }

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	// drawManual( frame );
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}
