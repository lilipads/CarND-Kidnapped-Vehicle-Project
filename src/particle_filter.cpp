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
#include <limits>

#include "particle_filter.h"

using namespace std;

const int NUM_PARTICLES = 100;
const double EPSILON = 0.0001;

std::random_device rd;
std::mt19937 gen(rd());

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = NUM_PARTICLES;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// initialize particles at random positions and theta drawn from normal distributions
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; i++) {
		// when yaw_rate is close to zero, use a different set of formula to prevent division by 0
		if (fabs(yaw_rate) < EPSILON){
			particles[i].x += velocity * cos(particles[i].theta) * delta_t;
			particles[i].y += velocity * sin(particles[i].theta) * delta_t;
		}
		else{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t)
				- sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t)
				+ cos(particles[i].theta));
			particles[i].theta += yaw_rate * delta_t;
		}
		// add noise to prediction
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// sense the location of all the map landmarks for each particle within its sensor_range
	sense(sensor_range, map_landmarks);

	// update the weight as the product of gaussians
	weights = {};
	for (auto &p : particles){
		// a list of observations as the nearest neighbors to each sensed landmark of the particle
		vector<LandmarkObs> ordered_obs = dataAssociation(observations, p); 
		p.weight = 1.;
		for (int i = 0; i < p.sense_x.size(); i++){
			p.weight *= multi_norm_pdf(
				ordered_obs[i].x, ordered_obs[i].y, p.sense_x[i], p.sense_y[i], 
				std_landmark[0], std_landmark[1]);
		}
		weights.push_back(p.weight);
	}
}

void ParticleFilter::sense(double sensor_range, const Map &map_landmarks){
	for (auto &p : particles){
		p.associations = {};
		p.sense_x = {};
		p.sense_y = {};
		// sense all landmarks within sensor_range
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
			if (dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f,
					p.x, p.y) < sensor_range){
				p.associations.push_back(map_landmarks.landmark_list[j].id_i);
				p.sense_x.push_back(map_landmarks.landmark_list[j].x_f);
				p.sense_y.push_back(map_landmarks.landmark_list[j].y_f);
			}
		}
	}
}

vector<LandmarkObs> ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& observations,
	Particle p) {
	std::vector<LandmarkObs> ordered_obs;
	std::vector<double> observations_x_map = {};
	std::vector<double> observations_y_map = {};
	for (auto &o : observations){
		// transform from vehicle coordinates to map coordinates
		observations_x_map.push_back(p.x + o.x * cos(p.theta) - o.y * sin(p.theta));
		observations_y_map.push_back(p.y + o.x * sin(p.theta) + o.y * cos(p.theta));
	}
	// for each landmark that is wihtin the sensor range of  the particle, 
	// associate it with an observation sensed by the vehicle
	for (int i = 0; i < p.sense_x.size(); i++){
		double min_dist = std::numeric_limits<double>::infinity();
		LandmarkObs best_obs;
		for (int j = 0; j < observations_x_map.size(); j++){
			double distance = dist(observations_x_map[j], observations_y_map[j],
				p.sense_x[i], p.sense_y[i]);
			if (distance < min_dist){
				min_dist = distance;
				best_obs.x = observations_x_map[j];
				best_obs.y = observations_y_map[j];
			}
		}
		ordered_obs.push_back(best_obs);
	}
	return ordered_obs;
}

void ParticleFilter::resample() {
	// draw particle with probability proportional to its weight
	std::discrete_distribution<int> d(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; i++){
		resampled_particles.push_back(particles[d(gen)]);
	}
	particles = resampled_particles;
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
