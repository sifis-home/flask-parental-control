# WP4 Analytic: Privacy-Aware Parental Control

[![Actions Status][actions badge]][actions]
[![CodeCov][codecov badge]][codecov]
[![LICENSE][license badge]][license]

<!-- Links -->
[actions]: https://github.com/sifis-home/flask-parental-control/actions
[codecov]: https://codecov.io/gh/sifis-home/flask-parental-control
[license]: LICENSES/MIT.txt

<!-- Badges -->
[actions badge]: https://github.com/sifis-home/flask-parental-control/workflows/flask-parental-control/badge.svg
[codecov badge]: https://codecov.io/gh/sifis-home/flask-parental-control/branch/master/graph/badge.svg
[license badge]: https://img.shields.io/badge/license-MIT-blue.svg

The parental control analytic takes as input data the video frames produced by the surveillance cameras distributed into the controlled environment, such as the camera of the controlled device or the surveillance cameras deployed in the environment, and the metadata provided by the controlled devices. The combination of multiple cameras allows to create a global view of the monitored environment, thus improving detection accuracy. The metadata about devices allows to understand the policy associated to such devices, which could express conditions to be met in order to be allowed to use the device, e.g., the function f of the device can be performed only by an adult user. 

The SIFIS-Home parental control analytic is composed by a pipeline of two components: 
- **Face Extraction Layer (FEL)**.
- **Age Classification Layer (ACL)**.

The age estimation model uses a [Keras implementation](https://github.com/yu4u/age-gender-estimation) that has been trained on the [IMDP-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), but to increase the accuracy of the model,we have classified age ranges into categories of "Toddler" for age value less than 13 years, "Child" for age value between 13 and 19, "Teen" for age value between 20 and 60, "Adult", and "Senior" for age value more than 60.

To preserve the privacy of data, Gaussian blurring is used for image/frame anonymization. Gaussian blurring convolves an image with a Gaussian filter of different sizes for anonymization. The privacy degree is controlled using the radius of blur metric. Gaussian blurring uses a low-pass filter which performs the smoothing function of an image. 

## Deploying

### Privacy-Aware Parental Control in a container
The DHT and the Analytics-API containers should be running before starting to build and run the image and container of the Privacy-Aware Parental Control.

Privacy-Aware Parental Control is intended to run in a docker container on port 6060. The Dockerfile at the root of this repo describes the container. To build and run it execute the following commands:

`docker build -t flask-parental-control .`

`docker-compose up`

## REST API of Privacy-Aware Parental Control

Description of the REST endpoint available while Privacy-Aware Parental Control is running.

---

#### GET /file_estimation

Description: Returns the result of whether the face in an image or a video frame is identified or not and the identity.

Command: 

`curl -F "file=@file_location" http://localhost:6060/file_estimation/<file_name.mp4>/<privacy_parameter>/<requestor_id>/<requestor_type>/<request_id>`

Sample: 

`curl -F "file=@file_location" http://localhost:6060/file_estimation/sample.mp4/17/33466553786f48cb72faad7b2fb9d0952c97/NSSD/2023061906001633466553786f48cb72faad7b2fb9d0952c97`

---
## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This software has been developed in the scope of the H2020 project SIFIS-Home with GA n. 952652.
