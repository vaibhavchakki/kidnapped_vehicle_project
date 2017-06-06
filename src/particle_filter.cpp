/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	weights.resize(num_particles);

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle P;
		P.id = i;
		P.x  = dist_x(gen);
		P.y  = dist_y(gen);
		P.theta = dist_theta(gen);
		P.weight = 1.0f;
		weights[i] = 1.0f;

		particles.push_back(P);
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// random Gaussuan noise
	default_random_engine gen;
	normal_distribution<double> noise_x(0.0f, std_pos[0]);
	normal_distribution<double> noise_y(0.0f, std_pos[1]);
	normal_distribution<double> noise_theta(0.0f, std_pos[2]);

	// adjust for 0 value to avoid divide by 0 error
	if (fabs(yaw_rate) < 0.0001) {
		yaw_rate = 0.0001;
	}

	for (int i = 0; i < num_particles; i++) {
		double v = velocity;
		double dt = delta_t;
		double y = yaw_rate;
		double th = particles[i].theta;

		particles[i].x += (v / y) * (sin(th + y * dt) - sin(th)) + noise_x(gen);
		particles[i].y += (v / y) * (cos(th) - cos(th + y * dt)) + noise_y(gen); 
		particles[i].theta += (y * dt) + noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int id;
	for(int i = 0; i < observations.size(); i++) {
		double min_distance = INT_MAX;

		for (int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < min_distance) {
				id = predicted[j].id;
				min_distance = distance;
			}
		}
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double x_var = std_landmark[0] * std_landmark[0];
	double y_var = std_landmark[1] * std_landmark[1];
	double xy_cov = std_landmark[0] * std_landmark[1];

	for (int i = 0; i < num_particles; i++) {

		// transform observations
		vector<LandmarkObs> transform_observation;

		double x_i = particles[i].x;
		double y_i = particles[i].y;
		double th_i = particles[i].theta;

		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs O;
			O.x = (observations[j].x * cos(th_i)) - (observations[j].y * sin(th_i)) + x_i;
			O.y = (observations[j].x * sin(th_i)) + (observations[j].y * cos(th_i)) + y_i;
			transform_observation.push_back(O);
		}

		// transform predict
		vector<LandmarkObs> transform_predict;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double distance = dist(x_i, y_i,
				                   map_landmarks.landmark_list[j].x_f,
				                   map_landmarks.landmark_list[j].y_f);

			if (distance < sensor_range) {
				LandmarkObs P;
				P.x = map_landmarks.landmark_list[j].x_f;
				P.y = map_landmarks.landmark_list[j].y_f;
				P.id = map_landmarks.landmark_list[j].id_i;
				transform_predict.push_back(P);
			}
		}

		// Associate data between observation and predict
		dataAssociation(transform_predict, transform_observation);

		double weight = 1;

		for (int j = 0; j < transform_predict.size(); j++) {
			double min_distance = INT_MAX;
			int min_idx = -1;

			for (int k = 0; k < transform_observation.size(); k++) {
				if (transform_observation[k].id == transform_predict[j].id) {
					double distance = dist(transform_predict[j].x, 
						                   transform_predict[j].y,
						                   transform_observation[k].x,
						                   transform_observation[k].y);

					if (distance < min_distance) {
						min_distance = distance;
						min_idx = k;
					}
				}
			}

			if (min_idx != -1) {
				double x_diff = transform_predict[j].x - transform_observation[min_idx].x;
				double y_diff = transform_predict[j].y - transform_observation[min_idx].y;

				double num = exp(-0.5 * ((x_diff * x_diff/x_var) + (y_diff* y_diff/y_var)));
				double den = 2 * M_PI * xy_cov;

				weight *= (num / den);
			}
		}

		weights[i] = weight;
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;

	discrete_distribution<> discrete_weights(weights.begin(), weights.end());
	vector<Particle> resampled_particles;

	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[discrete_weights(gen)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
