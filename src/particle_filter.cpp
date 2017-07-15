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
  // cout << endl << "init(x:" << x << " y:" << y << " theta:" << theta << " std:" << std[0] << "/" << std[1] << ")" << endl;

  num_particles = 20;
  weights.clear();
  particles.clear();

  static default_random_engine gen;
  normal_distribution<double> xdist(x,std[0]);
  normal_distribution<double> ydist(y,std[1]);
  normal_distribution<double> thetadist(theta,std[2]);

  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = xdist(gen);
    p.y = ydist(gen);
    p.theta = thetadist(gen);
    p.weight = 1.0/num_particles;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    particles.push_back(p);
    weights.push_back(p.weight);
    // cout << "  |p" << p.id << "| x:" << p.x << " y:" << p.y << " theta:" << p.theta << endl;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // cout << endl << "prediction(dt:" << delta_t << " v:" << velocity << " thetad:" << yaw_rate << " std:" << std_pos[0] << "/" << std_pos[1] << ")" << endl;


  static default_random_engine gen;
  normal_distribution<double> v_dist(velocity,std_pos[0]);
  normal_distribution<double> thetad_dist(yaw_rate,std_pos[1]);
  for(int i = 0; i < num_particles; i++) {
    double v = v_dist(gen);
    double thetad = thetad_dist(gen);
    particles[i].x += (v / thetad) * (sin(particles[i].theta + thetad * delta_t) - sin(particles[i].theta));
    particles[i].y += (- v / thetad) * (cos(particles[i].theta + thetad * delta_t) - cos(particles[i].theta));
    particles[i].theta += thetad * delta_t;
    // cout << "  |p" << particles[i].id << "| x:" << particles[i].x << " y:" << particles[i].y << " theta:" << particles[i].theta
    //      << " (v:" << v << " thetad:" << thetad << ")" << endl;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Unused function. Left in place so that project compiles without changing particle_filter.h
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  double std_range = std_landmark[0];
  double std_bearing = std_landmark[1];
  double range_squared = pow(sensor_range*1.2 + (10*std_range), 2);

  // cout << endl << "updateWeights(sensor_range:" << sensor_range << " std_range:" << std_range << " std_bearing:" << std_bearing << ")" << endl;

  vector<double> observation_distances;
  vector<double> observation_bearings;
  observation_distances.clear();
  observation_bearings.clear();
  for(int i = 0; i < observations.size(); i++) {
    LandmarkObs o = observations[i];
    observation_distances.push_back(sqrt(pow(o.x,2) + pow(o.y,2)));
    observation_bearings.push_back(atan2(o.y,o.x));
  }

  double total_weight = 0.0;

  for(int i = 0; i < num_particles; i++) {
    Particle part = particles[i];
    particles[i].weight = 1.0;
    particles[i].associations.clear();

    vector<Map::single_landmark_s> part_landmarks;
    vector<double> landmark_distances;
    vector<double> landmark_bearings;
    part_landmarks.clear();
    landmark_distances.clear();
    landmark_bearings.clear();
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s lm = map_landmarks.landmark_list[j];
      double lm_distance_squared = pow(part.x - lm.x_f, 2) + pow(part.y - lm.y_f, 2);
      if(range_squared >= lm_distance_squared) {
        part_landmarks.push_back(lm);
        // Represent landmarks as distance and bearing with respect to the particle.
        landmark_distances.push_back(sqrt(lm_distance_squared));
        double landmark_bearing = atan2(lm.y_f - part.y, lm.x_f - part.x) - part.theta;
        if(landmark_bearing > M_PI) landmark_bearing -= 2 * M_PI;
        if(landmark_bearing < -M_PI) landmark_bearing += 2 * M_PI;
        landmark_bearings.push_back(landmark_bearing);
      }
    }

    // cout << "  |p" << part.id << "| x:" << part.x << " y:" << part.y << " theta:" << part.theta << endl;
    for(int j = 0; j < observations.size(); j++) {
      int closest_id = -1;
      double closest_dist_stdev = 15.0;
      for(int k = 0; k < part_landmarks.size(); k++) {
        double dist_stdev = sqrt(pow((observation_distances[j] - landmark_distances[k]) / std_range, 2) +
                                 pow((observation_bearings[j] - landmark_bearings[k]) / std_bearing, 2));
        if(dist_stdev < closest_dist_stdev) {
          closest_dist_stdev = dist_stdev;
          closest_id = part_landmarks[k].id_i;
        }
      }
      particles[i].weight *= exp(-0.5 * closest_dist_stdev);
      if(closest_id > -1) {
        particles[i].associations.push_back(closest_id);
      }
    }
    total_weight += particles[i].weight;
  }
  weights.clear();
  for(int i = 0; i < num_particles; i++) {
    particles[i].weight = particles[i].weight / total_weight; // = 1.0 to give all particles equal weight for resample
    weights.push_back(particles[i].weight);
  }
}

void ParticleFilter::resample() {
  static default_random_engine gen;
  discrete_distribution<> p_dist(weights.begin(),weights.end());
  vector<Particle> new_particles;
  new_particles.clear();
  for(int i = 0; i < num_particles; i++) {
    Particle selected = particles[p_dist(gen)];
    Particle p;
    p.id = i;
    p.theta = selected.theta;
    p.x = selected.x;
    p.y = selected.y;
    p.weight = selected.weight; // set to i/200.0 to show whatever random particle is at end
    p.associations = selected.associations;
    p.sense_x = selected.sense_x;
    p.sense_y = selected.sense_y;
    new_particles.push_back(p);
  }
  particles = new_particles;

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
