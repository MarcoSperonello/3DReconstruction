#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace cv;

bool ReadInput(string, vector<Mat>&, vector<Mat>&, Mat&, Mat&, Size&, float&,
               int&);
bool ReadMatricesFromFile(Mat&, Mat&, Mat&, Mat&, Mat&, Mat&);
void CalibrateStereoCameras(vector<Mat>, vector<Mat>, Size, float, Mat&, Mat&,
                            Mat&, Mat&, Mat&, Mat&);
void UndistortAndRectify(Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat&, Mat&,
                         Mat&);
void ComputeDisparityMap(Mat, Mat, Mat&);
void ComputeDepthMap(Mat, Mat, Mat&);
void ViewReconstructedScene(Mat, Mat);
vector<vector<Point2f> > FindImagePoints(vector<Mat>, Size);
vector<vector<Point3f> > FindObjectPoints(vector <Mat>, Size, float);
void CalibrateSingleCamera(vector<vector<Point3f> >&, vector<vector<Point2f> >,
                           Size, Mat&, Mat&, vector<Mat>&, vector<Mat>&);
void ShowImages(vector<Mat>);

int main(int argc, char** argv) {
  vector<Mat> left_calibration_images, right_calibration_images;
  Mat left_image, right_image;
  Size chessboard_size;
  float square_size;
  int load_matrices;

  // Reads input data from input file
  string input_file = argc > 1 ? argv[1] : "input.xml";
  if (!ReadInput(input_file, left_calibration_images, right_calibration_images,
          left_image, right_image, chessboard_size, square_size, load_matrices))
    return 0;

  Mat camera_matrix_left = Mat::eye(3, 3, CV_64F);
  Mat camera_matrix_right = Mat::eye(3, 3, CV_64F);
  Mat dist_coeffs_left = Mat::zeros(8, 1, CV_64F);
  Mat dist_coeffs_right = Mat::zeros(8, 1, CV_64F);
  Mat R, T;

  // Calibrates the stereo set or reads the previously computed matrices from
  // file
  if (load_matrices == 0) {
    CalibrateStereoCameras(left_calibration_images, right_calibration_images,
                           chessboard_size, square_size, camera_matrix_left,
                           camera_matrix_right, dist_coeffs_left,
                           dist_coeffs_right, R, T);
  } else {
    ReadMatricesFromFile(R, T, camera_matrix_left, camera_matrix_right,
                         dist_coeffs_left, dist_coeffs_right);
  }

  /*cout << "camera_matrix_left: " << camera_matrix_left << endl;
  cout << "camera_matrix_right: " << camera_matrix_right << endl;
  cout << "dist_coeffs_left: " << dist_coeffs_left << endl;
  cout << "dist_coeffs_right: " << dist_coeffs_right << endl;*/

  // Undistorts and rectifies the two input pictures
  Mat undistorted_rectified_left, undistorted_rectified_right, Q;
  UndistortAndRectify(left_image, right_image, camera_matrix_left,
                      camera_matrix_right, dist_coeffs_left, dist_coeffs_right,
                      R, T, undistorted_rectified_left,
                      undistorted_rectified_right, Q);

  // Computes the disparity map
  Mat disparity_8;
  ComputeDisparityMap(undistorted_rectified_left, undistorted_rectified_right,
                      disparity_8);

  // Computes the depth map
  Mat depth_map;
  ComputeDepthMap(disparity_8, Q, depth_map);

  // Reconstructs and views the pointcloud
  ViewReconstructedScene(undistorted_rectified_left, depth_map);
    
  return 0;
}

// Reads input values from the XML input file.
// Returns true if data could successfully be read.
// The user must take care not to modify the names of the XML fields.
bool ReadInput(string file_name, vector<Mat>& left_calibration_images,
               vector<Mat>& right_calibration_images, Mat& left_image,
               Mat& right_image, Size& chessboard_size, float& square_size,
               int& load_matrices) {
  FileStorage fs(file_name, FileStorage::READ);

  if (!fs.isOpened()) {
    cout << "Failed to open file" << endl;
    return false;
  }

  // Number of inner chessboard corners per row/column
  int rows, cols;
  fs["rows"] >> rows;
  fs["columns"] >> cols;
  chessboard_size = Size(cols, rows);

  // Defines whether calibration matrices should be loaded from an XML file
  fs["load-matrices"] >> load_matrices;

  // Length of a chessboard square
  fs["square-size"] >> square_size;

  // Loads calibration images taken by left camera
  FileNode images = fs["calibration-images-left"];
  FileNodeIterator it = images.begin(), it_end = images.end();
  int i = 0;
  for (; it != it_end; ++it, i++) {
    left_calibration_images.push_back(imread((string) * it));
  }

  // Loads calibration images taken by right camera
  images = fs["calibration-images-right"];
  it = images.begin(), it_end = images.end();
  i = 0;
  for (; it != it_end; ++it, i++) {
    right_calibration_images.push_back(imread((string) * it));
  }

  // Loads left and right input images
  string image_path;
  fs["input-image-left"] >> image_path;
  left_image = imread(image_path);
  fs["input-image-right"] >> image_path;
  right_image = imread(image_path);

  fs.release();

  return true;
}

// Reads calibration matrices from matrices.xml.
// Returns true if data could successfully be read.
bool ReadMatricesFromFile(Mat& R, Mat& T,
                          Mat& camera_matrix_left, Mat& camera_matrix_right,
                          Mat& dist_coeffs_left, Mat& dist_coeffs_right) {
  FileStorage fs("matrices.xml", FileStorage::READ);

  if (!fs.isOpened()) {
    cout << "Failed to open file" << endl;
    return false;
  }

  fs["R"] >> R;
  fs["T"] >> T;
  fs["left-camera-matrix"] >> camera_matrix_left;
  fs["right-camera-matrix"] >> camera_matrix_right;
  fs["left-dist-coeffs"] >> dist_coeffs_left;
  fs["right-dist-coeffs"] >> dist_coeffs_right;

  fs.release();

  cout << "Matrices successfully read from file" << endl;

  return true;
}

// Views each of the input pictures in a window.
void ShowImages(vector<Mat> input) {
  for (int i = 0; i < input.size(); i++) {
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", input.at(i));
    waitKey(0);
  }
}

// Finds the image points in each of the calibration images.
vector<vector<Point2f> > FindImagePoints(vector<Mat> input, Size pattern_size) {
  const unsigned char kNumberOfImages = input.size();
  
  Mat gray;
  vector<vector<Point2f> > corners(kNumberOfImages);

  for (int i = 0; i < kNumberOfImages; i++) {
    cvtColor(input.at(i), gray, CV_BGR2GRAY);

    // Determines whether there is a chessboard pattern in the image and,
    // if present, the location of the internal chessboard corners
    //cout << "Searching corners for image #" << i << endl;
    bool pattern_found = findChessboardCorners(gray, pattern_size, corners.at(i),
                                               CALIB_CB_ADAPTIVE_THRESH +
                                               CALIB_CB_NORMALIZE_IMAGE +
                                               CALIB_CB_FAST_CHECK);

    // If a corner pattern is found, computes corner locations with
    // increased (subpixel) accuracy
    if (pattern_found) {
      cornerSubPix(gray, corners.at(i), Size(11, 11), Size(-1, -1),
                   TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    } else {
      corners.pop_back();
    }
  }
  
  return corners;
}

// Initializes the object points - the coordinates of each corner of the
// chessboard in the calibration pattern coordinate space. Returns a vector of
// vectors of points, one for each input image.
// This function works under the convenient assumption that the chessboard was
// kept still and the camera moved while capturing the scene.
// This means that the object points are the same for each of the input
// calibration images.
vector<vector<Point3f> > FindObjectPoints(vector<Mat> input, Size pattern_size,
                                          float square_size) {
  const unsigned char kNumberOfImages = input.size();

  // Computes the object points
  vector<Point3f> object_points;
  for (int i = 0; i < pattern_size.height; i++) {
    for (int j = 0; j < pattern_size.width; j++) {
      object_points.push_back(Point3f(float(j * square_size), 
                              float(i * square_size), 0));
    }
  }

  vector<vector<Point3f> > output;
  for (int i = 0; i < kNumberOfImages; i++) {
    output.push_back(object_points);
  }

  return output;
}

// Calibrates a single camera, computing the intrinsic matrix and the distortion
// coefficients matrix.
void CalibrateSingleCamera(vector<vector<Point3f> >& object_points,
                           vector<vector<Point2f> > image_points,
                           Size image_size, Mat& camera_matrix,
                           Mat& dist_coeffs, vector<Mat>& rvecs,
                           vector<Mat>& tvecs) {
  // Ensures the objectPoints vector of vectors is of the same size as the
  // imagePoints one, and fills it with objectPoints
  object_points.resize(image_points.size(), object_points.at(0));

  // Calibrates the camera and returns the average reprojection error of the
  // calibration
  double rms = calibrateCamera(object_points, image_points, image_size,
                                    camera_matrix, dist_coeffs, rvecs, tvecs);
  cout << "Camera successfully calibrated. rms: " << rms << endl;
}

// Calibrates the stereo camera pair, computing the intrinsic and distortion
// matrices of each camera as well as the rotation matrix and translation vector
// between the two cameras' coordinate systems.
void CalibrateStereoCameras(vector<Mat> left_images, vector<Mat> right_images,
                            Size chessboard_size, float square_size,
                            Mat& camera_matrix_left, Mat& camera_matrix_right,
                            Mat& dist_coeffs_left, Mat& dist_coeffs_right,
                            Mat& R, Mat& T) {
  // Computes image points for each camera
  cout << "Computing corners for right camera..." << endl;
  vector<vector<Point2f> > image_points_right =
    FindImagePoints(right_images, chessboard_size);
  cout << "Computing corners for left camera..." << endl;
  vector<vector<Point2f> > image_points_left =
    FindImagePoints(left_images, chessboard_size);

  // Computes object points
  vector<vector<Point3f> > object_points = 
    FindObjectPoints(right_images, chessboard_size, square_size);
  Size size = left_images.at(0).size();

  // Calibrates the two cameras separately
  vector<Mat> rvecs, tvecs;
  cout << endl << "Calibrating right camera..." << endl;
  CalibrateSingleCamera(object_points, image_points_right, size,
                        camera_matrix_right, dist_coeffs_right, rvecs, tvecs);
  rvecs.clear();
  tvecs.clear();
  cout << endl << "Calibrating left camera..." << endl;
  CalibrateSingleCamera(object_points, image_points_left, size,
                        camera_matrix_left, dist_coeffs_left, rvecs, tvecs);

  // Calibrates the stereo camera set, calculating the reprojection error
  cout << endl << "Calibrating stereo set..." << endl;
  Mat E, F;
  double rms = stereoCalibrate(object_points, image_points_left,
                               image_points_right, camera_matrix_left,
                               dist_coeffs_left, camera_matrix_right,
                               dist_coeffs_right, size, R, T, E, F);
  cout << "Stereo set successfully calibrated. rms: " << rms << endl << endl;

  // Saves the matrices in a file
  FileStorage fs("matrices.xml", FileStorage::WRITE);
  fs << "R" << R;
  fs << "T" << T;
  fs << "left-camera-matrix" << camera_matrix_left;
  fs << "right-camera-matrix" << camera_matrix_right;
  fs << "left-dist-coeffs" << dist_coeffs_left;
  fs << "right-dist-coeffs" << dist_coeffs_right;
  fs.release();
}

// Undistorts and rectifies the input images, one for each camera.
void UndistortAndRectify(Mat left_image, Mat right_image,
                         Mat camera_matrix_left, Mat camera_matrix_right,
                         Mat dist_coeffs_left, Mat dist_coeffs_right, Mat R,
                         Mat T, Mat& undistorted_rectified_left,
                         Mat& undistorted_rectified_right, Mat& Q) {
  // Flag for stereoRectify, ensures only valid pixels are visible after
  // rectification
  const double kAlpha = 0;
  
  // Computes rectification transforms, projection matrices and
  // disparity-to-depth mapping matrix for both cameras
  Size size = left_image.size();
  Mat R1, R2, P1, P2;
  stereoRectify(camera_matrix_left, dist_coeffs_left, camera_matrix_right,
                dist_coeffs_right, size, R, T, R1, R2, P1, P2, Q,
                CALIB_ZERO_DISPARITY, kAlpha);

  // Computes the undistortion and rectification transformation maps for
  // each camera
  Mat left_map_1, left_map_2, right_map_1, right_map_2;
  initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, R1, P1, size,
                          CV_32FC1, left_map_1, left_map_2);
  initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, size,
                          CV_32FC1, right_map_1, right_map_2);     

  // Applies the undistortion and rectification transformation maps to the
  // input images
  remap(left_image, undistorted_rectified_left, left_map_1, left_map_2,
        INTER_LINEAR);
  remap(right_image, undistorted_rectified_right, right_map_1, right_map_2,
        INTER_LINEAR);
  
}

// Computes the disparity map from the undistorted and rectified input images,
// one for each camera.
void ComputeDisparityMap(Mat undistorted_rectified_left,
                         Mat undistorted_rectified_right, Mat& disparity_8) {
  const char kImageChannels = 3;
  
  // Converts the input images to grayscale
  Mat gray_left, gray_right, disparity;
  cvtColor(undistorted_rectified_left, gray_left, CV_BGR2GRAY);
  cvtColor(undistorted_rectified_right, gray_right, CV_BGR2GRAY);

  // Initializes the object used to compute the stereo correspondence via the
  // semi-global block matching algorithm
  StereoSGBM stereo = StereoSGBM();
  stereo.numberOfDisparities = 320;
  stereo.SADWindowSize = 3;
  stereo.P1 = 8*kImageChannels*stereo.SADWindowSize*stereo.SADWindowSize*1.25;
  stereo.P2 = 32*kImageChannels*stereo.SADWindowSize*stereo.SADWindowSize*1.25;
  stereo.disp12MaxDiff = 10;

  // Computes the disparity map
  cout << "Computing disparity map..." << endl << endl;
  stereo(gray_left, gray_right, disparity);

  // Normalizes the disparity map
  normalize(disparity, disparity_8, 0, 255, CV_MINMAX, CV_8U);

  // Shows the disparity map
  namedWindow("Disparity map", WINDOW_NORMAL);
  imshow("Disparity map", disparity_8);
  waitKey(0);
  
}

// Computes the depth map from the disparity map and the disparity-to-depth
// mapping matrix.
void ComputeDepthMap(Mat disparity_8, Mat Q, Mat& depth_map) {
  // Computes the depth map
  cout << "Computing depth map..." << endl << endl;
  reprojectImageTo3D(disparity_8, depth_map, Q);

  // Shows the depth map
  namedWindow("Depth map", WINDOW_NORMAL);
  imshow("Depth map", depth_map);
  waitKey(0);
  
}

// Reconstructs and shows the 3D scene in the form of a point cloud from an
// undistorted rectified image and a depth map.
void ViewReconstructedScene(Mat image, Mat depth_map) {
  // Threshold value for XYZ coordinates of the points in the pointcloud
  const short kThreshold = 5000;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB output_point;

  // Builds the pointcloud
  for (int i = 0; i < depth_map.rows; i++) {
    for (int j = 0; j < depth_map.cols; j++) {
      Point3f point = depth_map.at<Point3f>(i, j);
      //if (disparity.at<uchar>(i, j) == 0) continue;
      Vec3b point_color = image.at<Vec3b>(i, j); //BGR 

      // Sets RGB values and XYZ coordinates of the point
      output_point.r = point_color[2];
      output_point.g = point_color[1];
      output_point.b = point_color[0];
      output_point.x = -point.x;
      output_point.y = -point.y;
      output_point.z = point.z;

      // If the XYZ absolute values aren't higher than the threshold,
      // the point is pushed into the pointcloud
      if (!(output_point.x > abs(kThreshold) || output_point.y > abs(kThreshold)
              || output_point.z > abs(kThreshold)))
        output->points.push_back(output_point);
    }
  }

  // Initializes the point cloud viewer and views the point cloud
  cout << "Visualization..." << endl;
  pcl::visualization::PCLVisualizer viewer("Reconstructed scene");
  viewer.setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
      output);
  viewer.addPointCloud<pcl::PointXYZRGB> (output, rgb, "cloud");
  viewer.addCoordinateSystem(1.0, "cloud");
  viewer.initCameraParameters();
  viewer.spin();
}
