#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>

using namespace cv;
using namespace std;

bool readInput(char*, vector<Mat>&, vector<Mat>&, Mat&, Mat&, Size&, float&, int&);
bool readMatricesFromFile(char*, Mat&, Mat&, Mat&, Mat&, Mat&, Mat&);
void calibrateStereoCameras(vector<Mat>, vector<Mat>, Size, float, Mat&, Mat&,
        Mat&, Mat&, Mat&, Mat&);
void undistortAndRectify(Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat&, Mat&, Mat&);
void computeDisparityMap(Mat, Mat, Mat&);
void computeDepthMap(Mat, Mat, Mat&);
void viewReconstructedScene(Mat,Mat);
vector<vector<Point2f> > findImagePoints(vector<Mat>, Size);
vector<vector<Point3f> > findObjectPoints(vector <Mat>, Size, float);
void calibrateSingleCamera(vector<vector<Point3f> >&, vector<vector<Point2f> >,
        Size, Mat&, Mat&, vector<Mat>&, vector<Mat>&);
void showPics(vector<Mat>);
double computeReprojectionErrors(const vector<vector<Point3f> >&,
        const vector<vector<Point2f> >&,
        const vector<Mat>&, const vector<Mat>&,
        const Mat&, const Mat&,
        vector<float>&);

int main(int argc, char** argv) {

  vector<Mat> leftCalibrationImages, rightCalibrationImages;
  Mat leftImage, rightImage;
  Size chessboardSize;
  float squareSize;
  int loadMatrices;


  if (!readInput(argv[1], leftCalibrationImages, rightCalibrationImages,
          leftImage, rightImage, chessboardSize, squareSize, loadMatrices))
    return 0;

  Mat cameraMatrix_left = Mat::eye(3, 3, CV_64F);
  Mat cameraMatrix_right = Mat::eye(3, 3, CV_64F);
  Mat distCoeffs_left = Mat::zeros(8, 1, CV_64F);
  Mat distCoeffs_right = Mat::zeros(8, 1, CV_64F);
  Mat R, T;

  if (loadMatrices == 0) {
    calibrateStereoCameras(leftCalibrationImages, rightCalibrationImages,
            chessboardSize, squareSize, cameraMatrix_left, cameraMatrix_right,
            distCoeffs_left, distCoeffs_right, R, T);
  } else {
    readMatricesFromFile(argv[1], R, T, cameraMatrix_left, cameraMatrix_right,
            distCoeffs_left, distCoeffs_right);
  }

  cout << "cameraMatrix_left: " << cameraMatrix_left << endl;
  cout << "cameraMatrix_right: " << cameraMatrix_right << endl;
  cout << "distCoeffs_left: " << distCoeffs_left << endl;
  cout << "distCoeffs_right: " << distCoeffs_right << endl;

  Mat undistortedRectified_left, undistortedRectified_right, Q;
  undistortAndRectify(leftImage, rightImage, cameraMatrix_left,
          cameraMatrix_right, distCoeffs_left, distCoeffs_right, R, T,
          undistortedRectified_left, undistortedRectified_right, Q);

  Mat disparity8;
  computeDisparityMap(undistortedRectified_left, undistortedRectified_right,
          disparity8);
  
  Mat depthMap;
  computeDepthMap(disparity8,Q,depthMap);

  viewReconstructedScene(undistortedRectified_left,depthMap);
          
  return 0;
}

bool readInput(char* fileName, vector<Mat>& leftCalibrationImages,
        vector<Mat>& rightCalibrationImages, Mat& leftImage, Mat& rightImage,
        Size& chessboardSize, float& squareSize, int& loadMatrices) {

  FileStorage fs(fileName, FileStorage::READ);

  if (!fs.isOpened()) {
    cout << "Failed to open file" << endl;
    return false;
  }

  int rows, cols;
  fs["rows"] >> rows;
  fs["columns"] >> cols;
  chessboardSize = Size(cols, rows);

  fs["load-matrices"] >> loadMatrices;

  fs["square-size"] >> squareSize;

  FileNode images = fs["calibration-images-left"];
  FileNodeIterator it = images.begin(), it_end = images.end();
  int i = 0;
  for (; it != it_end; ++it, i++) {
    leftCalibrationImages.push_back(imread((string) * it));
  }

  images = fs["calibration-images-right"];
  it = images.begin(), it_end = images.end();
  i = 0;
  for (; it != it_end; ++it, i++) {
    rightCalibrationImages.push_back(imread((string) * it));
  }

  string imagePath;
  fs["input-image-left"] >> imagePath;
  leftImage = imread(imagePath);
  fs["input-image-right"] >> imagePath;
  rightImage = imread(imagePath);

  fs.release();

  return true;
}

bool readMatricesFromFile(char* fileName, Mat& R, Mat& T,
        Mat& cameraMatrix_left, Mat& cameraMatrix_right, Mat& distCoeffs_left,
        Mat& distCoeffs_right) {
  FileStorage fs("matrices.xml", FileStorage::READ);

  if (!fs.isOpened()) {
    cout << "Failed to open file" << endl;
    return false;
  }

  fs["R"] >> R;
  fs["T"] >> T;
  fs["left-camera-matrix"] >> cameraMatrix_left;
  fs["right-camera-matrix"] >> cameraMatrix_right;
  fs["left-dist-coeffs"] >> distCoeffs_left;
  fs["right-dist-coeffs"] >> distCoeffs_right;

  fs.release();

  cout << "Matrices successfully read from file" << endl;

  return true;
}

void showPics(vector<Mat> input) {
  for (int i = 0; i < input.size(); i++) {
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", input.at(i));
    waitKey(0);
  }
}

vector<vector<Point2f> > findImagePoints(vector<Mat> input, Size patternSize) {
  const unsigned char IMG_NUMBER = input.size();

  Mat gray;
  vector<vector<Point2f> > corners(IMG_NUMBER);

  for (int i = 0; i < IMG_NUMBER; i++) {
    cvtColor(input.at(i), gray, CV_RGB2GRAY);

    cout << "Searching corners for image #" << i << endl;
    bool patternFound = findChessboardCorners(gray, patternSize, corners.at(i),
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if (patternFound) {
      cout << "Corners found. Polishing corners..." << endl << endl;
      cornerSubPix(gray, corners.at(i), Size(11, 11), Size(-1, -1),
              TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    } else {
      cout << "Corners not found." << endl << endl;
      corners.pop_back();
    }
    /*if(patternFound) {
      drawChessboardCorners(input.at(i), patternSize, Mat(corners.at(i)),
            patternFound);

      namedWindow("Corners", WINDOW_NORMAL);
      imshow("Corners", input.at(i));
      waitKey(0);
    }*/
  }

  return corners;
}

vector<vector<Point3f> > findObjectPoints(vector<Mat> input, Size patternSize,
        float squareSize) {
  const unsigned char IMG_NUMBER = input.size();

  vector<Point3f> objectPoints;
  for (int i = 0; i < patternSize.height; i++) {
    for (int j = 0; j < patternSize.width; j++) {
      objectPoints.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
    }
  }

  vector<vector<Point3f> > output;
  for (int i = 0; i < IMG_NUMBER; i++) {
    output.push_back(objectPoints);
  }

  return output;
}

void calibrateSingleCamera(vector<vector<Point3f> >& objectPoints,
        vector<vector<Point2f> > imagePoints, Size imageSize, Mat& cameraMatrix,
        Mat& distCoeffs, vector<Mat>& rvecs, vector<Mat>& tvecs) {

  objectPoints.resize(imagePoints.size(), objectPoints.at(0));
  cout << objectPoints.size() << endl;

  bool successful = calibrateCamera(objectPoints, imagePoints, imageSize,
          cameraMatrix, distCoeffs, rvecs, tvecs);

  if (successful) {
    cout << "Camera successfully calibrated" << endl;
  } else {
    cout << "Couldn't calibrate camera" << endl;
  }

  cout << "cameraMatrix: " << endl;
  cout << cameraMatrix << endl;

  cout << "distCoeffs: " << endl;
  cout << distCoeffs << endl;

  vector<float> perViewErrors;
  double rms = computeReprojectionErrors(objectPoints, imagePoints, rvecs,
          tvecs, cameraMatrix, distCoeffs, perViewErrors);
  cout << "rms: " << rms << endl;
  /*for (int i = 0; i < perViewErrors.size(); i++) {
    cout << "perViewErrors for image #" << i << ": " << perViewErrors.at(i) << endl;
  }*/

}

void calibrateStereoCameras(vector<Mat> leftImages, vector<Mat> rightImages,
        Size chessboardSize, float squareSize, Mat& cameraMatrix_left,
        Mat& cameraMatrix_right, Mat& distCoeffs_left, Mat& distCoeffs_right,
        Mat& R, Mat& T) {

  vector<vector<Point2f> > imagePoints_right =
          findImagePoints(rightImages, chessboardSize);
  vector<vector<Point2f> > imagePoints_left =
          findImagePoints(leftImages, chessboardSize);
  vector<vector<Point3f> > objectPoints = findObjectPoints(rightImages,
          chessboardSize, squareSize);
  Size size = leftImages.at(0).size();

  vector<Mat> rvecs, tvecs;
  calibrateSingleCamera(objectPoints, imagePoints_right, size,
          cameraMatrix_right, distCoeffs_right, rvecs, tvecs);
  rvecs.clear();
  tvecs.clear();
  calibrateSingleCamera(objectPoints, imagePoints_left, size, cameraMatrix_left,
          distCoeffs_left, rvecs, tvecs);

  Mat E, F;
  double rms = stereoCalibrate(objectPoints, imagePoints_left, imagePoints_right,
          cameraMatrix_left, distCoeffs_left, cameraMatrix_right,
          distCoeffs_right, size, R, T, E, F);
  cout << "Stereo rms: " << rms << endl;

  FileStorage fs("matrices.xml", FileStorage::WRITE);
  fs << "R" << R;
  fs << "T" << T;
  fs << "left-camera-matrix" << cameraMatrix_left;
  fs << "right-camera-matrix" << cameraMatrix_right;
  fs << "left-dist-coeffs" << distCoeffs_left;
  fs << "right-dist-coeffs" << distCoeffs_right;
  fs.release();
}

void undistortAndRectify(Mat leftImage, Mat rightImage, Mat cameraMatrix_left,
        Mat cameraMatrix_right, Mat distCoeffs_left, Mat distCoeffs_right,
        Mat R, Mat T, Mat& undistortedRectified_left,
        Mat& undistortedRectified_right, Mat& Q) {

  Size size = leftImage.size();
  Mat R1, R2, P1, P2;
  stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right,
          distCoeffs_right, size, R, T, R1, R2, P1, P2, Q);

  Mat leftMap1, leftMap2, rightMap1, rightMap2;
  initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, size,
          CV_32FC1, leftMap1, leftMap2);
  initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, size,
          CV_32FC1, rightMap1, rightMap2);

  remap(leftImage, undistortedRectified_left,
          leftMap1, leftMap2, INTER_LINEAR);
  remap(rightImage, undistortedRectified_right,
          rightMap1, rightMap2, INTER_LINEAR);
}

void computeDisparityMap(Mat undistortedRectified_left,
        Mat undistortedRectified_right, Mat& disparity8) {

  Mat gray_left, gray_right, disparity;
  cvtColor(undistortedRectified_left, gray_left, CV_BGR2GRAY);
  cvtColor(undistortedRectified_right, gray_right, CV_BGR2GRAY);

  //StereoSGBM stereo = StereoSGBM(-64,192,5);
  //StereoSGBM stereo = StereoSGBM(-64,192,5,600,2400,10,4,1,150,2,false);
  int n1, n2, n3;
  n1 = 0;
  n2 = 272;
  n3 = 3;
  StereoSGBM stereo = StereoSGBM(n1, n2, n3);
  stereo.operator()(gray_left, gray_right, disparity);
  normalize(disparity, disparity8, 0, 255, CV_MINMAX, CV_8U);

  namedWindow("Disparity map", WINDOW_NORMAL);
  imshow("Disparity map", disparity8);
  waitKey(0);

  char temp[200];
  sprintf(temp, "../../../disparity %d,%d,%d.png", n1, n2, n3);
  imwrite(temp, disparity8);
}

void computeDepthMap(Mat disparity8, Mat Q, Mat& depthMap) {
  //Mat depthMap = Mat(disparity8.size(), CV_32FC3);
  reprojectImageTo3D(disparity8, depthMap, Q);

  namedWindow("Depth map", WINDOW_NORMAL);
  imshow("Depth map", depthMap);
  waitKey(0);
}

void viewReconstructedScene(Mat image, Mat depthMap) {
  const short THRESHOLD = 5000;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB outputPoint;
  for (int i = 0; i < depthMap.rows; i++) {
    for (int j = 0; j < depthMap.cols; j++) {
      Point3f point = depthMap.at<Point3f>(i, j);
      //if (disparity.at<uchar>(i, j) == 0) continue;
      Vec3b pointColor = image.at<Vec3b>(i, j); //BGR 

      outputPoint.r = pointColor[2];
      outputPoint.g = pointColor[1];
      outputPoint.b = pointColor[0];
      outputPoint.x = point.x;
      outputPoint.y = -point.y;
      outputPoint.z = point.z;
      if (!(outputPoint.x > abs(THRESHOLD) || outputPoint.y > abs(THRESHOLD)
              || outputPoint.z > abs(THRESHOLD)))
        output->points.push_back(outputPoint);
    }
  }
  
  pcl::visualization::PCLVisualizer viewer("Reconstructed scene");

  viewer.setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(output);
  viewer.addPointCloud<pcl::PointXYZRGB> (output, rgb, "cloud");
  viewer.addCoordinateSystem(1.0, "cloud");
  viewer.initCameraParameters();
  viewer.spin();
}

double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors) {

  vector<Point2f> imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());
  for (i = 0; i < (int) objectPoints.size(); ++i) {
    projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, // project
            distCoeffs, imagePoints2);
    err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2); // difference
    int n = (int) objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err * err / n); // save for this view
    totalErr += err*err; // sum it up
    totalPoints += n;
  }

  return sqrt(totalErr / totalPoints); // calculate the arithmetical mean
}
