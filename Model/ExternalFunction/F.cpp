#include <OpenSim/Simulation/Model/Model.h>
#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>
#include <OpenSim/Simulation/SimbodyEngine/Joint.h>
#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>
#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>
#include <OpenSim/Common/LinearFunction.h>
#include <OpenSim/Common/PolynomialFunction.h>
#include <OpenSim/Common/MultiplierFunction.h>
#include <OpenSim/Common/Constant.h>
#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>
#include <OpenSim/Simulation/SimulationUtilities.h>
#include "SimTKcommon/internal/recorder.h"

#include <iostream>
#include <iterator>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <fstream>

using namespace SimTK;
using namespace OpenSim;

constexpr int n_in = 2; 
constexpr int n_out = 1; 
constexpr int nCoordinates = 33; 
constexpr int NX = nCoordinates*2; 
constexpr int NU = nCoordinates; 
constexpr int NR = 177; 

template<typename T> 
T value(const Recorder& e) { return e; }; 
template<> 
double value(const Recorder& e) { return e.getValue(); }; 

template<typename T>
int F_generic(const T** arg, T** res) {

	// Definition of model.
	OpenSim::Model* model;
	model = new OpenSim::Model();

	// Definition of bodies.
	OpenSim::Body* pelvis;
	pelvis = new OpenSim::Body("pelvis", 12.53720482631376498261, Vec3(-0.08452177196387893054, 0.00000000000000000000, 0.00000000000000000000), Inertia(0.15187603322578829834, 0.15187603322578829834, 0.08809331820525076462, 0., 0., 0.));
	model->addBody(pelvis);

	OpenSim::Body* femur_r;
	femur_r = new OpenSim::Body("femur_r", 9.90180495639592805901, Vec3(0.00000000000000000000, -0.17105206679665946323, 0.00000000000000000000), Inertia(0.14431299254860657788, 0.03782961940594541800, 0.15218069117149551905, 0., 0., 0.));
	model->addBody(femur_r);

	OpenSim::Body* tibia_r;
	tibia_r = new OpenSim::Body("tibia_r", 3.94681896013910904486, Vec3(0.00000000000000000000, -0.19052685224849860712, 0.00000000000000000000), Inertia(0.05587535964457789051, 0.00565405424974895366, 0.05665140630630813301, 0., 0., 0.));
	model->addBody(tibia_r);

	OpenSim::Body* talus_r;
	talus_r = new OpenSim::Body("talus_r", 0.10645499555331379427, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Inertia(0.00097029790329154796, 0.00097029790329154796, 0.00097029790329154796, 0., 0., 0.));
	model->addBody(talus_r);

	OpenSim::Body* calcn_r;
	calcn_r = new OpenSim::Body("calcn_r", 1.33068744441642250464, Vec3(0.09547057173478119829, 0.02864117152043435879, 0.00000000000000000000), Inertia(0.00135841706460816712, 0.00378416182283703708, 0.00397822140349534618, 0., 0., 0.));
	model->addBody(calcn_r);

	OpenSim::Body* toes_r;
	toes_r = new OpenSim::Body("toes_r", 0.23058152036847764932, Vec3(0.03303281782023428925, 0.00572823430408687159, -0.01670735005358671074), Inertia(0.00009702979032915480, 0.00019405958065830961, 0.00097029790329154796, 0., 0., 0.));
	model->addBody(toes_r);

	OpenSim::Body* femur_l;
	femur_l = new OpenSim::Body("femur_l", 9.90180495639592805901, Vec3(0.00000000000000000000, -0.17038044996004714227, 0.00000000000000000000), Inertia(0.14318195965772231859, 0.03753313505590779448, 0.15098799629328149563, 0., 0., 0.));
	model->addBody(femur_l);

	OpenSim::Body* tibia_l;
	tibia_l = new OpenSim::Body("tibia_l", 3.94681896013910904486, Vec3(0.00000000000000000000, -0.18865096122766922271, 0.00000000000000000000), Inertia(0.05478049994762054786, 0.00554326487565207950, 0.05554134022467083209, 0., 0., 0.));
	model->addBody(tibia_l);

	OpenSim::Body* talus_l;
	talus_l = new OpenSim::Body("talus_l", 0.10645499555331379427, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Inertia(0.00097029790329154796, 0.00097029790329154796, 0.00097029790329154796, 0., 0., 0.));
	model->addBody(talus_l);

	OpenSim::Body* calcn_l;
	calcn_l = new OpenSim::Body("calcn_l", 1.33068744441642250464, Vec3(0.09547057173478119829, 0.02864117152043435879, 0.00000000000000000000), Inertia(0.00135841706460816712, 0.00378416182283703708, 0.00397822140349534618, 0., 0., 0.));
	model->addBody(calcn_l);

	OpenSim::Body* toes_l;
	toes_l = new OpenSim::Body("toes_l", 0.23058152036847764932, Vec3(0.03303281782023428925, 0.00572823430408687159, 0.01670735005358671074), Inertia(0.00009702979032915480, 0.00019405958065830961, 0.00097029790329154796, 0., 0., 0.));
	model->addBody(toes_l);

	OpenSim::Body* torso;
	torso = new OpenSim::Body("torso", 28.55825583710527837411, Vec3(-0.03000776421943438660, 0.32008281834063345705, 0.00000000000000000000), Inertia(1.59623588765048651794, 0.85617261451838722053, 1.59623588765048651794, 0., 0., 0.));
	model->addBody(torso);

	OpenSim::Body* humerus_r;
	humerus_r = new OpenSim::Body("humerus_r", 2.16369778462110318173, Vec3(0.00000000000000000000, -0.16163108780716814450, 0.00000000000000000000), Inertia(0.01227710537252689124, 0.00423522109829091876, 0.01378065510967797433, 0., 0., 0.));
	model->addBody(humerus_r);

	OpenSim::Body* ulna_r;
	ulna_r = new OpenSim::Body("ulna_r", 0.64671409798638135324, Vec3(0.00000000000000000000, -0.12919869239436143626, 0.00000000000000000000), Inertia(0.00362337310581214945, 0.00075599074253609327, 0.00393041788959298984, 0., 0., 0.));
	model->addBody(ulna_r);

	OpenSim::Body* radius_r;
	radius_r = new OpenSim::Body("radius_r", 0.64671409798638135324, Vec3(0.00000000000000000000, -0.12919869239436143626, 0.00000000000000000000), Inertia(0.00362337310581214945, 0.00075599074253609327, 0.00393041788959298984, 0., 0., 0.));
	model->addBody(radius_r);

	OpenSim::Body* hand_r;
	hand_r = new OpenSim::Body("hand_r", 0.48703160465641059940, Vec3(0.00000000000000000000, -0.07299551925819575127, 0.00000000000000000000), Inertia(0.00109117110411358445, 0.00066913743716382357, 0.00163920322815269432, 0., 0., 0.));
	model->addBody(hand_r);

	OpenSim::Body* humerus_l;
	humerus_l = new OpenSim::Body("humerus_l", 2.16369778462110318173, Vec3(0.00000000000000000000, -0.16163108780716814450, 0.00000000000000000000), Inertia(0.01227710537252689124, 0.00423522109829091876, 0.01378065510967797433, 0., 0., 0.));
	model->addBody(humerus_l);

	OpenSim::Body* ulna_l;
	ulna_l = new OpenSim::Body("ulna_l", 0.64671409798638135324, Vec3(0.00000000000000000000, -0.12919869239436143626, 0.00000000000000000000), Inertia(0.00362337310581214945, 0.00075599074253609327, 0.00393041788959298984, 0., 0., 0.));
	model->addBody(ulna_l);

	OpenSim::Body* radius_l;
	radius_l = new OpenSim::Body("radius_l", 0.64671409798638135324, Vec3(0.00000000000000000000, -0.12919869239436143626, 0.00000000000000000000), Inertia(0.00362337310581214945, 0.00075599074253609327, 0.00393041788959298984, 0., 0., 0.));
	model->addBody(radius_l);

	OpenSim::Body* hand_l;
	hand_l = new OpenSim::Body("hand_l", 0.48703160465641059940, Vec3(0.00000000000000000000, -0.07299551925819575127, 0.00000000000000000000), Inertia(0.00109117110411358445, 0.00066913743716382357, 0.00163920322815269432, 0., 0., 0.));
	model->addBody(hand_l);

	// Definition of joints.
	SpatialTransform st_ground_pelvis;
	st_ground_pelvis[0].setCoordinateNames(OpenSim::Array<std::string>("pelvis_tilt", 1, 1));
	st_ground_pelvis[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_ground_pelvis[1].setCoordinateNames(OpenSim::Array<std::string>("pelvis_list", 1, 1));
	st_ground_pelvis[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[2].setCoordinateNames(OpenSim::Array<std::string>("pelvis_rotation", 1, 1));
	st_ground_pelvis[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[3].setCoordinateNames(OpenSim::Array<std::string>("pelvis_tx", 1, 1));
	st_ground_pelvis[3].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[4].setCoordinateNames(OpenSim::Array<std::string>("pelvis_ty", 1, 1));
	st_ground_pelvis[4].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_ground_pelvis[5].setCoordinateNames(OpenSim::Array<std::string>("pelvis_tz", 1, 1));
	st_ground_pelvis[5].setFunction(new LinearFunction(1.0000, 0.0000));
	st_ground_pelvis[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* ground_pelvis;
	ground_pelvis = new OpenSim::CustomJoint("ground_pelvis", model->getGround(), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *pelvis, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_ground_pelvis);
	model->addJoint(ground_pelvis);

	SpatialTransform st_hip_r;
	st_hip_r[0].setCoordinateNames(OpenSim::Array<std::string>("hip_flexion_r", 1, 1));
	st_hip_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_r[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_hip_r[1].setCoordinateNames(OpenSim::Array<std::string>("hip_adduction_r", 1, 1));
	st_hip_r[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_r[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_r[2].setCoordinateNames(OpenSim::Array<std::string>("hip_rotation_r", 1, 1));
	st_hip_r[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_r[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.19549889623591143462));
	st_hip_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.19549889623591143462));
	st_hip_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.17114392327415584028));
	st_hip_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* hip_r;
	hip_r = new OpenSim::CustomJoint("hip_r", *pelvis, Vec3(-0.06727789588457215586, -0.09383470836555669881, 0.09048257951216127914), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *femur_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_hip_r);
	model->addJoint(hip_r);

	SpatialTransform st_walker_knee_r;
	st_walker_knee_r[0].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	st_walker_knee_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_walker_knee_r[0].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_r[1].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_walker_knee_r_1_coeffs[5] = {0.01083209453986300023, -0.02521832550124099986, -0.03284781039885199816, 0.07910001196702699799, -0.00000001473252350900}; 
	Vector st_walker_knee_r_1_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_r_1_coeffs_vec[i] = st_walker_knee_r_1_coeffs[i]; 
	st_walker_knee_r[1].setFunction(new PolynomialFunction(st_walker_knee_r_1_coeffs_vec));
	st_walker_knee_r[1].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_walker_knee_r[2].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_walker_knee_r_2_coeffs[4] = {0.02516576272742300155, -0.16948005139054000967, 0.36949934868824901857, -0.00000004430358308836}; 
	Vector st_walker_knee_r_2_coeffs_vec(4); 
	for (int i = 0; i < 4; ++i) st_walker_knee_r_2_coeffs_vec[i] = st_walker_knee_r_2_coeffs[i]; 
	st_walker_knee_r[2].setFunction(new PolynomialFunction(st_walker_knee_r_2_coeffs_vec));
	st_walker_knee_r[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_r[3].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_walker_knee_r_3_coeffs[5] = {0.00015904478788503811, -0.00101514991566899995, 0.00181751097496800004, 0.00002641426645199230, -0.00000077465635324719}; 
	Vector st_walker_knee_r_3_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_r_3_coeffs_vec[i] = st_walker_knee_r_3_coeffs[i]; 
	st_walker_knee_r[3].setFunction(new MultiplierFunction(new PolynomialFunction(st_walker_knee_r_3_coeffs_vec), 1.00618862821564380994));
	st_walker_knee_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_r[4].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_walker_knee_r_4_coeffs[5] = {-0.00057968780523386836, 0.00507976574562600015, -0.01144237572636399962, 0.00393690866884400040, -0.00002516350383213525}; 
	Vector st_walker_knee_r_4_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_r_4_coeffs_vec[i] = st_walker_knee_r_4_coeffs[i]; 
	st_walker_knee_r[4].setFunction(new MultiplierFunction(new PolynomialFunction(st_walker_knee_r_4_coeffs_vec), 1.00618862821564380994));
	st_walker_knee_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_r[5].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_r", 1, 1));
	osim_double_adouble st_walker_knee_r_5_coeffs[5] = {0.00120808688920599999, -0.00445361122470600002, 0.00061164940729817395, 0.00626542960638699995, -0.00001461912533723326}; 
	Vector st_walker_knee_r_5_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_r_5_coeffs_vec[i] = st_walker_knee_r_5_coeffs[i]; 
	st_walker_knee_r[5].setFunction(new MultiplierFunction(new PolynomialFunction(st_walker_knee_r_5_coeffs_vec), 1.00618862821564380994));
	st_walker_knee_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* walker_knee_r;
	walker_knee_r = new OpenSim::CustomJoint("walker_knee_r", *femur_r, Vec3(-0.00452784882697039679, -0.41213486211712774265, -0.00176083009937737667), Vec3(-1.64156999999999997364, 1.44618000000000002103, 1.57079999999999997407), *tibia_r, Vec3(-0.00825535400288333714, -0.00360725397604709595, -0.00151517321160919011), Vec3(-1.64156999999999997364, 1.44618000000000002103, 1.57079999999999997407), st_walker_knee_r);
	model->addJoint(walker_knee_r);

	OpenSim::PinJoint* ankle_r;
	ankle_r = new OpenSim::PinJoint("ankle_r", *tibia_r, Vec3(-0.01020497333950179907, -0.40819893358007197648, 0.00000000000000000000), Vec3(0.17589499999999999580, -0.10520799999999999597, 0.01866220000000000032), *talus_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.17589499999999999580, -0.10520799999999999597, 0.01866220000000000032));
	model->addJoint(ankle_r);

	OpenSim::PinJoint* subtalar_r;
	subtalar_r = new OpenSim::PinJoint("subtalar_r", *talus_r, Vec3(-0.04656099783505279061, -0.04004990484274071477, 0.00756126928139467039), Vec3(-1.76818999999999992845, 0.90622300000000000075, 1.81960000000000010623), *calcn_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-1.76818999999999992845, 0.90622300000000000075, 1.81960000000000010623));
	model->addJoint(subtalar_r);

	OpenSim::PinJoint* mtp_r;
	mtp_r = new OpenSim::PinJoint("mtp_r", *calcn_r, Vec3(0.17070138226178876995, -0.00190941143469562386, 0.00103108217473563699), Vec3(-3.14158999999999988262, 0.61990100000000003533, 0.00000000000000000000), *toes_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-3.14158999999999988262, 0.61990100000000003533, 0.00000000000000000000));
	model->addJoint(mtp_r);

	SpatialTransform st_hip_l;
	st_hip_l[0].setCoordinateNames(OpenSim::Array<std::string>("hip_flexion_l", 1, 1));
	st_hip_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_hip_l[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_hip_l[1].setCoordinateNames(OpenSim::Array<std::string>("hip_adduction_l", 1, 1));
	st_hip_l[1].setFunction(new LinearFunction(-1.0000, 0.0000));
	st_hip_l[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_l[2].setCoordinateNames(OpenSim::Array<std::string>("hip_rotation_l", 1, 1));
	st_hip_l[2].setFunction(new LinearFunction(-1.0000, 0.0000));
	st_hip_l[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.19549889623591143462));
	st_hip_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_hip_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.19549889623591143462));
	st_hip_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_hip_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.17114392327415584028));
	st_hip_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* hip_l;
	hip_l = new OpenSim::CustomJoint("hip_l", *pelvis, Vec3(-0.06727789588457215586, -0.09383470836555669881, -0.09048257951216127914), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *femur_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_hip_l);
	model->addJoint(hip_l);

	SpatialTransform st_walker_knee_l;
	st_walker_knee_l[0].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	st_walker_knee_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_walker_knee_l[0].setAxis(Vec3(-1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_l[1].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_walker_knee_l_1_coeffs[5] = {0.01083209453986300023, -0.02521832550124099986, -0.03284781039885199816, 0.07910001196702699799, -0.00000001473252350900}; 
	Vector st_walker_knee_l_1_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_l_1_coeffs_vec[i] = st_walker_knee_l_1_coeffs[i]; 
	st_walker_knee_l[1].setFunction(new PolynomialFunction(st_walker_knee_l_1_coeffs_vec));
	st_walker_knee_l[1].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_walker_knee_l[2].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_walker_knee_l_2_coeffs[4] = {-0.02516576272742300155, 0.16948005139054000967, -0.36949934868824901857, 0.00000004430358308836}; 
	Vector st_walker_knee_l_2_coeffs_vec(4); 
	for (int i = 0; i < 4; ++i) st_walker_knee_l_2_coeffs_vec[i] = st_walker_knee_l_2_coeffs[i]; 
	st_walker_knee_l[2].setFunction(new PolynomialFunction(st_walker_knee_l_2_coeffs_vec));
	st_walker_knee_l[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_l[3].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_walker_knee_l_3_coeffs[5] = {0.00015904478788503811, -0.00101514991566899995, 0.00181751097496800004, 0.00002641426645199230, -0.00000077465635324719}; 
	Vector st_walker_knee_l_3_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_l_3_coeffs_vec[i] = st_walker_knee_l_3_coeffs[i]; 
	st_walker_knee_l[3].setFunction(new MultiplierFunction(new PolynomialFunction(st_walker_knee_l_3_coeffs_vec), 1.00223794094145368661));
	st_walker_knee_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_l[4].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_walker_knee_l_4_coeffs[5] = {-0.00057968780523386836, 0.00507976574562600015, -0.01144237572636399962, 0.00393690866884400040, -0.00002516350383213525}; 
	Vector st_walker_knee_l_4_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_l_4_coeffs_vec[i] = st_walker_knee_l_4_coeffs[i]; 
	st_walker_knee_l[4].setFunction(new MultiplierFunction(new PolynomialFunction(st_walker_knee_l_4_coeffs_vec), 1.00223794094145368661));
	st_walker_knee_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_walker_knee_l[5].setCoordinateNames(OpenSim::Array<std::string>("knee_angle_l", 1, 1));
	osim_double_adouble st_walker_knee_l_5_coeffs[5] = {-0.00120808688920599999, 0.00445361122470600002, -0.00061164940729817395, -0.00626542960638699995, 0.00001461912533723326}; 
	Vector st_walker_knee_l_5_coeffs_vec(5); 
	for (int i = 0; i < 5; ++i) st_walker_knee_l_5_coeffs_vec[i] = st_walker_knee_l_5_coeffs[i]; 
	st_walker_knee_l[5].setFunction(new MultiplierFunction(new PolynomialFunction(st_walker_knee_l_5_coeffs_vec), 1.00223794094145368661));
	st_walker_knee_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* walker_knee_l;
	walker_knee_l = new OpenSim::CustomJoint("walker_knee_l", *femur_l, Vec3(-0.00451007073423654106, -0.41051666060961944371, 0.00175391639664754392), Vec3(1.64156999999999997364, -1.44618000000000002103, 1.57079999999999997407), *tibia_l, Vec3(-0.00817407336309415762, -0.00357173764192589803, 0.00150025510537316335), Vec3(1.64156999999999997364, -1.44618000000000002103, 1.57079999999999997407), st_walker_knee_l);
	model->addJoint(walker_knee_l);

	OpenSim::PinJoint* ankle_l;
	ankle_l = new OpenSim::PinJoint("ankle_l", *tibia_l, Vec3(-0.01010449711985373505, -0.40417988479414940217, 0.00000000000000000000), Vec3(-0.17589499999999999580, 0.10520799999999999597, 0.01866220000000000032), *talus_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-0.17589499999999999580, 0.10520799999999999597, 0.01866220000000000032));
	model->addJoint(ankle_l);

	OpenSim::PinJoint* subtalar_l;
	subtalar_l = new OpenSim::PinJoint("subtalar_l", *talus_l, Vec3(-0.04656099783505279061, -0.04004990484274071477, -0.00756126928139467039), Vec3(1.76818999999999992845, -0.90622300000000000075, 1.81960000000000010623), *calcn_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(1.76818999999999992845, -0.90622300000000000075, 1.81960000000000010623));
	model->addJoint(subtalar_l);

	OpenSim::PinJoint* mtp_l;
	mtp_l = new OpenSim::PinJoint("mtp_l", *calcn_l, Vec3(0.17070138226178876995, -0.00190941143469562386, -0.00103108217473563699), Vec3(-3.14158999999999988262, -0.61990100000000003533, 0.00000000000000000000), *toes_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-3.14158999999999988262, -0.61990100000000003533, 0.00000000000000000000));
	model->addJoint(mtp_l);

	SpatialTransform st_back;
	st_back[0].setCoordinateNames(OpenSim::Array<std::string>("lumbar_extension", 1, 1));
	st_back[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_back[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_back[1].setCoordinateNames(OpenSim::Array<std::string>("lumbar_bending", 1, 1));
	st_back[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_back[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_back[2].setCoordinateNames(OpenSim::Array<std::string>("lumbar_rotation", 1, 1));
	st_back[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_back[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_back[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.19549889623591143462));
	st_back[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_back[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.19549889623591143462));
	st_back[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_back[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.17114392327415584028));
	st_back[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* back;
	back = new OpenSim::CustomJoint("back", *pelvis, Vec3(-0.12038673885095628469, 0.09743316004322678803, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *torso, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_back);
	model->addJoint(back);

	SpatialTransform st_acromial_r;
	st_acromial_r[0].setCoordinateNames(OpenSim::Array<std::string>("arm_flex_r", 1, 1));
	st_acromial_r[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_r[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_acromial_r[1].setCoordinateNames(OpenSim::Array<std::string>("arm_add_r", 1, 1));
	st_acromial_r[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_r[1].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[2].setCoordinateNames(OpenSim::Array<std::string>("arm_rot_r", 1, 1));
	st_acromial_r[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_r[2].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.00025880731447958105));
	st_acromial_r[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.00025880731447958105));
	st_acromial_r[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_acromial_r[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.06426170056973745659));
	st_acromial_r[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* acromial_r;
	acromial_r = new OpenSim::CustomJoint("acromial_r", *torso, Vec3(0.00315581653707718273, 0.37159614691732917313, 0.18092448909685537872), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *humerus_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_acromial_r);
	model->addJoint(acromial_r);

	OpenSim::PinJoint* elbow_r;
	elbow_r = new OpenSim::PinJoint("elbow_r", *humerus_r, Vec3(0.01291460905118124898, -0.28127692307583762599, -0.00942754670161929931), Vec3(-0.02286269999999999969, 0.22801799999999999846, 0.00516889999999999971), *ulna_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-0.02286269999999999969, 0.22801799999999999846, 0.00516889999999999971));
	model->addJoint(elbow_r);

	OpenSim::PinJoint* radioulnar_r;
	radioulnar_r = new OpenSim::PinJoint("radioulnar_r", *ulna_r, Vec3(-0.00721111473749736203, -0.01394306070917618150, 0.02796008706676730501), Vec3(-1.56884000000000001229, 0.05642799999999999899, 1.53614000000000006096), *radius_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-1.56884000000000001229, 0.05642799999999999899, 1.53614000000000006096));
	model->addJoint(radioulnar_r);

	OpenSim::WeldJoint* radius_hand_r;
	radius_hand_r = new OpenSim::WeldJoint("radius_hand_r", *radius_r, Vec3(-0.00943008418994563391, -0.25281351431635429305, 0.01458945615836763736), Vec3(-1.57079999999999997407, 0.00000000000000000000, -1.57079999999999997407), *hand_r, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(-1.57079999999999997407, 0.00000000000000000000, -1.57079999999999997407));
	model->addJoint(radius_hand_r);

	SpatialTransform st_acromial_l;
	st_acromial_l[0].setCoordinateNames(OpenSim::Array<std::string>("arm_flex_l", 1, 1));
	st_acromial_l[0].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_l[0].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	st_acromial_l[1].setCoordinateNames(OpenSim::Array<std::string>("arm_add_l", 1, 1));
	st_acromial_l[1].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_l[1].setAxis(Vec3(-1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[2].setCoordinateNames(OpenSim::Array<std::string>("arm_rot_l", 1, 1));
	st_acromial_l[2].setFunction(new LinearFunction(1.0000, 0.0000));
	st_acromial_l[2].setAxis(Vec3(0.00000000000000000000, -1.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[3].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.00025880731447958105));
	st_acromial_l[3].setAxis(Vec3(1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[4].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.00025880731447958105));
	st_acromial_l[4].setAxis(Vec3(0.00000000000000000000, 1.00000000000000000000, 0.00000000000000000000));
	st_acromial_l[5].setFunction(new MultiplierFunction(new Constant(0.00000000000000000000), 1.06426170056973745659));
	st_acromial_l[5].setAxis(Vec3(0.00000000000000000000, 0.00000000000000000000, 1.00000000000000000000));
	OpenSim::CustomJoint* acromial_l;
	acromial_l = new OpenSim::CustomJoint("acromial_l", *torso, Vec3(0.00315581653707718273, 0.37159614691732917313, -0.18092448909685537872), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), *humerus_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), st_acromial_l);
	model->addJoint(acromial_l);

	OpenSim::PinJoint* elbow_l;
	elbow_l = new OpenSim::PinJoint("elbow_l", *humerus_l, Vec3(0.01291460905118124898, -0.28127692307583762599, 0.00942754670161929931), Vec3(0.02286269999999999969, -0.22801799999999999846, 0.00516889999999999971), *ulna_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(0.02286269999999999969, -0.22801799999999999846, 0.00516889999999999971));
	model->addJoint(elbow_l);

	OpenSim::PinJoint* radioulnar_l;
	radioulnar_l = new OpenSim::PinJoint("radioulnar_l", *ulna_l, Vec3(-0.00721111473749736203, -0.01394306070917618150, -0.02796008706676730501), Vec3(1.56884000000000001229, -0.05642799999999999899, 1.53614000000000006096), *radius_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(1.56884000000000001229, -0.05642799999999999899, 1.53614000000000006096));
	model->addJoint(radioulnar_l);

	OpenSim::WeldJoint* radius_hand_l;
	radius_hand_l = new OpenSim::WeldJoint("radius_hand_l", *radius_l, Vec3(-0.00943008418994563391, -0.25281351431635429305, -0.01458945615836763736), Vec3(1.57079999999999997407, 0.00000000000000000000, 1.57079999999999997407), *hand_l, Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000), Vec3(1.57079999999999997407, 0.00000000000000000000, 1.57079999999999997407));
	model->addJoint(radius_hand_l);

	// Definition of contacts.
	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s1_r;
	SmoothSphereHalfSpaceForce_s1_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s1_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s1_r_location(0.00198599260060105696, -0.01000000000000000021, -0.00399704363882922350);
	SmoothSphereHalfSpaceForce_s1_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s1_r_location);
	double SmoothSphereHalfSpaceForce_s1_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s1_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s1_r_radius );
	SmoothSphereHalfSpaceForce_s1_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s1_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s1_r->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s1_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s1_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s1_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s1_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s2_r;
	SmoothSphereHalfSpaceForce_s2_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s2_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s2_r_location(0.15500779539802719831, -0.01000000000000000021, -0.02999469124160162453);
	SmoothSphereHalfSpaceForce_s2_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s2_r_location);
	double SmoothSphereHalfSpaceForce_s2_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s2_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s2_r_radius );
	SmoothSphereHalfSpaceForce_s2_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s2_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s2_r->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s2_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s2_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s2_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s2_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s3_r;
	SmoothSphereHalfSpaceForce_s3_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s3_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s3_r_location(0.13893603624864139801, -0.01000000000000000021, 0.05394039390870120770);
	SmoothSphereHalfSpaceForce_s3_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s3_r_location);
	double SmoothSphereHalfSpaceForce_s3_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s3_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s3_r_radius );
	SmoothSphereHalfSpaceForce_s3_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s3_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s3_r->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s3_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s3_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s3_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s3_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s4_r;
	SmoothSphereHalfSpaceForce_s4_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s4_r", *calcn_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s4_r_location(0.06919023300292503897, -0.01000000000000000021, 0.02754059957798050615);
	SmoothSphereHalfSpaceForce_s4_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s4_r_location);
	double SmoothSphereHalfSpaceForce_s4_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s4_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s4_r_radius );
	SmoothSphereHalfSpaceForce_s4_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s4_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s4_r->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s4_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s4_r->connectSocket_sphere_frame(*calcn_r);
	SmoothSphereHalfSpaceForce_s4_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s4_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s5_r;
	SmoothSphereHalfSpaceForce_s5_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s5_r", *toes_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s5_r_location(0.06267735943127515641, -0.01000000000000000021, -0.01959744327515347551);
	SmoothSphereHalfSpaceForce_s5_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s5_r_location);
	double SmoothSphereHalfSpaceForce_s5_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s5_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s5_r_radius );
	SmoothSphereHalfSpaceForce_s5_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s5_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s5_r->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s5_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s5_r->connectSocket_sphere_frame(*toes_r);
	SmoothSphereHalfSpaceForce_s5_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s5_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s6_r;
	SmoothSphereHalfSpaceForce_s6_r = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s6_r", *toes_r, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s6_r_location(0.04700801957345637078, -0.01000000000000000021, 0.06461717853092996400);
	SmoothSphereHalfSpaceForce_s6_r->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s6_r_location);
	double SmoothSphereHalfSpaceForce_s6_r_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s6_r->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s6_r_radius );
	SmoothSphereHalfSpaceForce_s6_r->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s6_r->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s6_r->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_r->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_r->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_r->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_r->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s6_r->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s6_r->connectSocket_sphere_frame(*toes_r);
	SmoothSphereHalfSpaceForce_s6_r->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s6_r);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s1_l;
	SmoothSphereHalfSpaceForce_s1_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s1_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s1_l_location(0.00198599260060105696, -0.01000000000000000021, 0.00399704363882922350);
	SmoothSphereHalfSpaceForce_s1_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s1_l_location);
	double SmoothSphereHalfSpaceForce_s1_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s1_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s1_l_radius );
	SmoothSphereHalfSpaceForce_s1_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s1_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s1_l->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s1_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s1_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s1_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s1_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s1_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s1_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s2_l;
	SmoothSphereHalfSpaceForce_s2_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s2_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s2_l_location(0.15500779539802719831, -0.01000000000000000021, 0.02999469124160162453);
	SmoothSphereHalfSpaceForce_s2_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s2_l_location);
	double SmoothSphereHalfSpaceForce_s2_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s2_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s2_l_radius );
	SmoothSphereHalfSpaceForce_s2_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s2_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s2_l->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s2_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s2_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s2_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s2_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s2_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s2_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s3_l;
	SmoothSphereHalfSpaceForce_s3_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s3_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s3_l_location(0.13893603624864139801, -0.01000000000000000021, -0.05394039390870120770);
	SmoothSphereHalfSpaceForce_s3_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s3_l_location);
	double SmoothSphereHalfSpaceForce_s3_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s3_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s3_l_radius );
	SmoothSphereHalfSpaceForce_s3_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s3_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s3_l->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s3_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s3_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s3_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s3_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s3_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s3_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s4_l;
	SmoothSphereHalfSpaceForce_s4_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s4_l", *calcn_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s4_l_location(0.06919023300292503897, -0.01000000000000000021, -0.02754059957798050615);
	SmoothSphereHalfSpaceForce_s4_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s4_l_location);
	double SmoothSphereHalfSpaceForce_s4_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s4_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s4_l_radius );
	SmoothSphereHalfSpaceForce_s4_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s4_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s4_l->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s4_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s4_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s4_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s4_l->connectSocket_sphere_frame(*calcn_l);
	SmoothSphereHalfSpaceForce_s4_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s4_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s5_l;
	SmoothSphereHalfSpaceForce_s5_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s5_l", *toes_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s5_l_location(0.06267735943127515641, -0.01000000000000000021, 0.01959744327515347551);
	SmoothSphereHalfSpaceForce_s5_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s5_l_location);
	double SmoothSphereHalfSpaceForce_s5_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s5_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s5_l_radius );
	SmoothSphereHalfSpaceForce_s5_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s5_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s5_l->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s5_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s5_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s5_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s5_l->connectSocket_sphere_frame(*toes_l);
	SmoothSphereHalfSpaceForce_s5_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s5_l);

	OpenSim::SmoothSphereHalfSpaceForce* SmoothSphereHalfSpaceForce_s6_l;
	SmoothSphereHalfSpaceForce_s6_l = new SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_s6_l", *toes_l, model->getGround());
	Vec3 SmoothSphereHalfSpaceForce_s6_l_location(0.04700801957345637078, -0.01000000000000000021, -0.06461717853092996400);
	SmoothSphereHalfSpaceForce_s6_l->set_contact_sphere_location(SmoothSphereHalfSpaceForce_s6_l_location);
	double SmoothSphereHalfSpaceForce_s6_l_radius = (0.03200000000000000067);
	SmoothSphereHalfSpaceForce_s6_l->set_contact_sphere_radius(SmoothSphereHalfSpaceForce_s6_l_radius );
	SmoothSphereHalfSpaceForce_s6_l->set_contact_half_space_location(Vec3(0.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000));
	SmoothSphereHalfSpaceForce_s6_l->set_contact_half_space_orientation(Vec3(0.00000000000000000000, 0.00000000000000000000, -1.57079632679489655800));
	SmoothSphereHalfSpaceForce_s6_l->set_stiffness(1000000.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_l->set_dissipation(2.00000000000000000000);
	SmoothSphereHalfSpaceForce_s6_l->set_static_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_l->set_dynamic_friction(0.80000000000000004441);
	SmoothSphereHalfSpaceForce_s6_l->set_viscous_friction(0.50000000000000000000);
	SmoothSphereHalfSpaceForce_s6_l->set_transition_velocity(0.20000000000000001110);
	SmoothSphereHalfSpaceForce_s6_l->connectSocket_sphere_frame(*toes_l);
	SmoothSphereHalfSpaceForce_s6_l->connectSocket_half_space_frame(model->getGround());
	model->addComponent(SmoothSphereHalfSpaceForce_s6_l);

	// Initialize system.
	SimTK::State* state;
	state = new State(model->initSystem());

	// Read inputs.
	std::vector<T> x(arg[0], arg[0] + NX);
	std::vector<T> u(arg[1], arg[1] + NU);

	// States and controls.
	T ua[NU];
	Vector QsUs(NX);
	/// States
	for (int i = 0; i < NX; ++i) QsUs[i] = x[i];
	/// Controls
	/// OpenSim and Simbody have different state orders.
	auto indicesOSInSimbody = getIndicesOpenSimInSimbody(*model);
	for (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];

	// Set state variables and realize.
	model->setStateVariableValues(*state, QsUs);
	model->realizeVelocity(*state);

	// Compute residual forces.
	/// Set appliedMobilityForces (# mobilities).
	Vector appliedMobilityForces(nCoordinates);
	appliedMobilityForces.setToZero();
	/// Set appliedBodyForces (# bodies + ground).
	Vector_<SpatialVec> appliedBodyForces;
	int nbodies = model->getBodySet().getSize() + 1;
	appliedBodyForces.resize(nbodies);
	appliedBodyForces.setToZero();
	/// Set gravity.
	Vec3 gravity(0);
	gravity[1] = -9.80664999999999942304;
	/// Add weights to appliedBodyForces.
	for (int i = 0; i < model->getBodySet().getSize(); ++i) {
		model->getMatterSubsystem().addInStationForce(*state,
		model->getBodySet().get(i).getMobilizedBodyIndex(),
		model->getBodySet().get(i).getMassCenter(),
		model->getBodySet().get(i).getMass()*gravity, appliedBodyForces);
	}
	/// Add contact forces to appliedBodyForces.
	Array<osim_double_adouble> Force_0 = SmoothSphereHalfSpaceForce_s1_r->getRecordValues(*state);
	SpatialVec GRF_0;
	GRF_0[0] = Vec3(Force_0[3], Force_0[4], Force_0[5]);
	GRF_0[1] = Vec3(Force_0[0], Force_0[1], Force_0[2]);
	int c_idx_0 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_0] += GRF_0;

	Array<osim_double_adouble> Force_1 = SmoothSphereHalfSpaceForce_s2_r->getRecordValues(*state);
	SpatialVec GRF_1;
	GRF_1[0] = Vec3(Force_1[3], Force_1[4], Force_1[5]);
	GRF_1[1] = Vec3(Force_1[0], Force_1[1], Force_1[2]);
	int c_idx_1 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_1] += GRF_1;

	Array<osim_double_adouble> Force_2 = SmoothSphereHalfSpaceForce_s3_r->getRecordValues(*state);
	SpatialVec GRF_2;
	GRF_2[0] = Vec3(Force_2[3], Force_2[4], Force_2[5]);
	GRF_2[1] = Vec3(Force_2[0], Force_2[1], Force_2[2]);
	int c_idx_2 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_2] += GRF_2;

	Array<osim_double_adouble> Force_3 = SmoothSphereHalfSpaceForce_s4_r->getRecordValues(*state);
	SpatialVec GRF_3;
	GRF_3[0] = Vec3(Force_3[3], Force_3[4], Force_3[5]);
	GRF_3[1] = Vec3(Force_3[0], Force_3[1], Force_3[2]);
	int c_idx_3 = model->getBodySet().get("calcn_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_3] += GRF_3;

	Array<osim_double_adouble> Force_4 = SmoothSphereHalfSpaceForce_s5_r->getRecordValues(*state);
	SpatialVec GRF_4;
	GRF_4[0] = Vec3(Force_4[3], Force_4[4], Force_4[5]);
	GRF_4[1] = Vec3(Force_4[0], Force_4[1], Force_4[2]);
	int c_idx_4 = model->getBodySet().get("toes_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_4] += GRF_4;

	Array<osim_double_adouble> Force_5 = SmoothSphereHalfSpaceForce_s6_r->getRecordValues(*state);
	SpatialVec GRF_5;
	GRF_5[0] = Vec3(Force_5[3], Force_5[4], Force_5[5]);
	GRF_5[1] = Vec3(Force_5[0], Force_5[1], Force_5[2]);
	int c_idx_5 = model->getBodySet().get("toes_r").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_5] += GRF_5;

	Array<osim_double_adouble> Force_6 = SmoothSphereHalfSpaceForce_s1_l->getRecordValues(*state);
	SpatialVec GRF_6;
	GRF_6[0] = Vec3(Force_6[3], Force_6[4], Force_6[5]);
	GRF_6[1] = Vec3(Force_6[0], Force_6[1], Force_6[2]);
	int c_idx_6 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_6] += GRF_6;

	Array<osim_double_adouble> Force_7 = SmoothSphereHalfSpaceForce_s2_l->getRecordValues(*state);
	SpatialVec GRF_7;
	GRF_7[0] = Vec3(Force_7[3], Force_7[4], Force_7[5]);
	GRF_7[1] = Vec3(Force_7[0], Force_7[1], Force_7[2]);
	int c_idx_7 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_7] += GRF_7;

	Array<osim_double_adouble> Force_8 = SmoothSphereHalfSpaceForce_s3_l->getRecordValues(*state);
	SpatialVec GRF_8;
	GRF_8[0] = Vec3(Force_8[3], Force_8[4], Force_8[5]);
	GRF_8[1] = Vec3(Force_8[0], Force_8[1], Force_8[2]);
	int c_idx_8 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_8] += GRF_8;

	Array<osim_double_adouble> Force_9 = SmoothSphereHalfSpaceForce_s4_l->getRecordValues(*state);
	SpatialVec GRF_9;
	GRF_9[0] = Vec3(Force_9[3], Force_9[4], Force_9[5]);
	GRF_9[1] = Vec3(Force_9[0], Force_9[1], Force_9[2]);
	int c_idx_9 = model->getBodySet().get("calcn_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_9] += GRF_9;

	Array<osim_double_adouble> Force_10 = SmoothSphereHalfSpaceForce_s5_l->getRecordValues(*state);
	SpatialVec GRF_10;
	GRF_10[0] = Vec3(Force_10[3], Force_10[4], Force_10[5]);
	GRF_10[1] = Vec3(Force_10[0], Force_10[1], Force_10[2]);
	int c_idx_10 = model->getBodySet().get("toes_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_10] += GRF_10;

	Array<osim_double_adouble> Force_11 = SmoothSphereHalfSpaceForce_s6_l->getRecordValues(*state);
	SpatialVec GRF_11;
	GRF_11[0] = Vec3(Force_11[3], Force_11[4], Force_11[5]);
	GRF_11[1] = Vec3(Force_11[0], Force_11[1], Force_11[2]);
	int c_idx_11 = model->getBodySet().get("toes_l").getMobilizedBodyIndex();
	appliedBodyForces[c_idx_11] += GRF_11;

	/// knownUdot.
	Vector knownUdot(nCoordinates);
	knownUdot.setToZero();
	for (int i = 0; i < nCoordinates; ++i) knownUdot[i] = ua[i];
	/// Calculate residual forces.
	Vector residualMobilityForces(nCoordinates);
	residualMobilityForces.setToZero();
	model->getMatterSubsystem().calcResidualForceIgnoringConstraints(*state,
			appliedMobilityForces, appliedBodyForces, knownUdot, residualMobilityForces);

	/// Body origins.
	Vec3 pelvis_or = pelvis->getPositionInGround(*state);
	Vec3 femur_r_or = femur_r->getPositionInGround(*state);
	Vec3 tibia_r_or = tibia_r->getPositionInGround(*state);
	Vec3 talus_r_or = talus_r->getPositionInGround(*state);
	Vec3 calcn_r_or = calcn_r->getPositionInGround(*state);
	Vec3 toes_r_or = toes_r->getPositionInGround(*state);
	Vec3 femur_l_or = femur_l->getPositionInGround(*state);
	Vec3 tibia_l_or = tibia_l->getPositionInGround(*state);
	Vec3 talus_l_or = talus_l->getPositionInGround(*state);
	Vec3 calcn_l_or = calcn_l->getPositionInGround(*state);
	Vec3 toes_l_or = toes_l->getPositionInGround(*state);
	Vec3 torso_or = torso->getPositionInGround(*state);
	Vec3 humerus_r_or = humerus_r->getPositionInGround(*state);
	Vec3 ulna_r_or = ulna_r->getPositionInGround(*state);
	Vec3 radius_r_or = radius_r->getPositionInGround(*state);
	Vec3 hand_r_or = hand_r->getPositionInGround(*state);
	Vec3 humerus_l_or = humerus_l->getPositionInGround(*state);
	Vec3 ulna_l_or = ulna_l->getPositionInGround(*state);
	Vec3 radius_l_or = radius_l->getPositionInGround(*state);
	Vec3 hand_l_or = hand_l->getPositionInGround(*state);

	/// Ground reaction forces.
	Vec3 GRF_r(0), GRF_l(0);
	GRF_r += GRF_0[1];
	GRF_r += GRF_1[1];
	GRF_r += GRF_2[1];
	GRF_r += GRF_3[1];
	GRF_r += GRF_4[1];
	GRF_r += GRF_5[1];
	GRF_l += GRF_6[1];
	GRF_l += GRF_7[1];
	GRF_l += GRF_8[1];
	GRF_l += GRF_9[1];
	GRF_l += GRF_10[1];
	GRF_l += GRF_11[1];

	/// Ground reaction moments.
	Vec3 GRM_r(0), GRM_l(0);
	Vec3 normal(0, 1, 0);

	SimTK::Transform TR_GB_calcn_r = calcn_r->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s1_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s1_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s1_r_locationCP_G = SmoothSphereHalfSpaceForce_s1_r_location_G - SmoothSphereHalfSpaceForce_s1_r_radius * normal;
	Vec3 locationCP_G_adj_0 = SmoothSphereHalfSpaceForce_s1_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s1_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s1_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_0, *calcn_r);
	Vec3 GRM_0 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s1_r_locationCP_B) % GRF_0[1];
	GRM_r += GRM_0;

	Vec3 SmoothSphereHalfSpaceForce_s2_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s2_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s2_r_locationCP_G = SmoothSphereHalfSpaceForce_s2_r_location_G - SmoothSphereHalfSpaceForce_s2_r_radius * normal;
	Vec3 locationCP_G_adj_1 = SmoothSphereHalfSpaceForce_s2_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s2_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s2_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_1, *calcn_r);
	Vec3 GRM_1 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s2_r_locationCP_B) % GRF_1[1];
	GRM_r += GRM_1;

	Vec3 SmoothSphereHalfSpaceForce_s3_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s3_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s3_r_locationCP_G = SmoothSphereHalfSpaceForce_s3_r_location_G - SmoothSphereHalfSpaceForce_s3_r_radius * normal;
	Vec3 locationCP_G_adj_2 = SmoothSphereHalfSpaceForce_s3_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s3_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s3_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_2, *calcn_r);
	Vec3 GRM_2 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s3_r_locationCP_B) % GRF_2[1];
	GRM_r += GRM_2;

	Vec3 SmoothSphereHalfSpaceForce_s4_r_location_G = calcn_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s4_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s4_r_locationCP_G = SmoothSphereHalfSpaceForce_s4_r_location_G - SmoothSphereHalfSpaceForce_s4_r_radius * normal;
	Vec3 locationCP_G_adj_3 = SmoothSphereHalfSpaceForce_s4_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s4_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s4_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_3, *calcn_r);
	Vec3 GRM_3 = (TR_GB_calcn_r*SmoothSphereHalfSpaceForce_s4_r_locationCP_B) % GRF_3[1];
	GRM_r += GRM_3;

	SimTK::Transform TR_GB_toes_r = toes_r->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s5_r_location_G = toes_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s5_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s5_r_locationCP_G = SmoothSphereHalfSpaceForce_s5_r_location_G - SmoothSphereHalfSpaceForce_s5_r_radius * normal;
	Vec3 locationCP_G_adj_4 = SmoothSphereHalfSpaceForce_s5_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s5_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s5_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_4, *toes_r);
	Vec3 GRM_4 = (TR_GB_toes_r*SmoothSphereHalfSpaceForce_s5_r_locationCP_B) % GRF_4[1];
	GRM_r += GRM_4;

	Vec3 SmoothSphereHalfSpaceForce_s6_r_location_G = toes_r->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s6_r_location);
	Vec3 SmoothSphereHalfSpaceForce_s6_r_locationCP_G = SmoothSphereHalfSpaceForce_s6_r_location_G - SmoothSphereHalfSpaceForce_s6_r_radius * normal;
	Vec3 locationCP_G_adj_5 = SmoothSphereHalfSpaceForce_s6_r_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s6_r_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s6_r_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_5, *toes_r);
	Vec3 GRM_5 = (TR_GB_toes_r*SmoothSphereHalfSpaceForce_s6_r_locationCP_B) % GRF_5[1];
	GRM_r += GRM_5;

	SimTK::Transform TR_GB_calcn_l = calcn_l->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s1_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s1_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s1_l_locationCP_G = SmoothSphereHalfSpaceForce_s1_l_location_G - SmoothSphereHalfSpaceForce_s1_l_radius * normal;
	Vec3 locationCP_G_adj_6 = SmoothSphereHalfSpaceForce_s1_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s1_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s1_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_6, *calcn_l);
	Vec3 GRM_6 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s1_l_locationCP_B) % GRF_6[1];
	GRM_l += GRM_6;

	Vec3 SmoothSphereHalfSpaceForce_s2_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s2_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s2_l_locationCP_G = SmoothSphereHalfSpaceForce_s2_l_location_G - SmoothSphereHalfSpaceForce_s2_l_radius * normal;
	Vec3 locationCP_G_adj_7 = SmoothSphereHalfSpaceForce_s2_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s2_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s2_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_7, *calcn_l);
	Vec3 GRM_7 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s2_l_locationCP_B) % GRF_7[1];
	GRM_l += GRM_7;

	Vec3 SmoothSphereHalfSpaceForce_s3_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s3_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s3_l_locationCP_G = SmoothSphereHalfSpaceForce_s3_l_location_G - SmoothSphereHalfSpaceForce_s3_l_radius * normal;
	Vec3 locationCP_G_adj_8 = SmoothSphereHalfSpaceForce_s3_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s3_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s3_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_8, *calcn_l);
	Vec3 GRM_8 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s3_l_locationCP_B) % GRF_8[1];
	GRM_l += GRM_8;

	Vec3 SmoothSphereHalfSpaceForce_s4_l_location_G = calcn_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s4_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s4_l_locationCP_G = SmoothSphereHalfSpaceForce_s4_l_location_G - SmoothSphereHalfSpaceForce_s4_l_radius * normal;
	Vec3 locationCP_G_adj_9 = SmoothSphereHalfSpaceForce_s4_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s4_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s4_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_9, *calcn_l);
	Vec3 GRM_9 = (TR_GB_calcn_l*SmoothSphereHalfSpaceForce_s4_l_locationCP_B) % GRF_9[1];
	GRM_l += GRM_9;

	SimTK::Transform TR_GB_toes_l = toes_l->getMobilizedBody().getBodyTransform(*state);
	Vec3 SmoothSphereHalfSpaceForce_s5_l_location_G = toes_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s5_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s5_l_locationCP_G = SmoothSphereHalfSpaceForce_s5_l_location_G - SmoothSphereHalfSpaceForce_s5_l_radius * normal;
	Vec3 locationCP_G_adj_10 = SmoothSphereHalfSpaceForce_s5_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s5_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s5_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_10, *toes_l);
	Vec3 GRM_10 = (TR_GB_toes_l*SmoothSphereHalfSpaceForce_s5_l_locationCP_B) % GRF_10[1];
	GRM_l += GRM_10;

	Vec3 SmoothSphereHalfSpaceForce_s6_l_location_G = toes_l->findStationLocationInGround(*state, SmoothSphereHalfSpaceForce_s6_l_location);
	Vec3 SmoothSphereHalfSpaceForce_s6_l_locationCP_G = SmoothSphereHalfSpaceForce_s6_l_location_G - SmoothSphereHalfSpaceForce_s6_l_radius * normal;
	Vec3 locationCP_G_adj_11 = SmoothSphereHalfSpaceForce_s6_l_locationCP_G - 0.5*SmoothSphereHalfSpaceForce_s6_l_locationCP_G[1] * normal;
	Vec3 SmoothSphereHalfSpaceForce_s6_l_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_11, *toes_l);
	Vec3 GRM_11 = (TR_GB_toes_l*SmoothSphereHalfSpaceForce_s6_l_locationCP_B) % GRF_11[1];
	GRM_l += GRM_11;

	/// Outputs.
	/// Residual forces (OpenSim and Simbody have different state orders).
	auto indicesSimbodyInOS = getIndicesSimbodyInOpenSim(*model);
	for (int i = 0; i < NU; ++i) res[0][i] =
			value<T>(residualMobilityForces[indicesSimbodyInOS[i]]);
	/// Ground reaction forces.
	for (int i = 0; i < 3; ++i) res[0][i + 33] = value<T>(GRF_r[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 36] = value<T>(GRF_l[i]);
	/// Ground reaction moments.
	for (int i = 0; i < 3; ++i) res[0][i + 39] = value<T>(GRM_r[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 42] = value<T>(GRM_l[i]);
	/// Ground reaction forces per sphere.
	for (int i = 0; i < 3; ++i) res[0][i + 45] = value<T>(GRF_0[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 48] = value<T>(GRF_1[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 51] = value<T>(GRF_2[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 54] = value<T>(GRF_3[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 57] = value<T>(GRF_4[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 60] = value<T>(GRF_5[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 63] = value<T>(GRF_6[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 66] = value<T>(GRF_7[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 69] = value<T>(GRF_8[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 72] = value<T>(GRF_9[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 75] = value<T>(GRF_10[1][i]);
	for (int i = 0; i < 3; ++i) res[0][i + 78] = value<T>(GRF_11[1][i]);

	/// Contact point locations per sphere.
	for (int i = 0; i < 3; ++i) res[0][i + 81] = value<T>(locationCP_G_adj_0[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 84] = value<T>(locationCP_G_adj_1[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 87] = value<T>(locationCP_G_adj_2[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 90] = value<T>(locationCP_G_adj_3[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 93] = value<T>(locationCP_G_adj_4[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 96] = value<T>(locationCP_G_adj_5[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 99] = value<T>(locationCP_G_adj_6[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 102] = value<T>(locationCP_G_adj_7[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 105] = value<T>(locationCP_G_adj_8[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 108] = value<T>(locationCP_G_adj_9[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 111] = value<T>(locationCP_G_adj_10[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 114] = value<T>(locationCP_G_adj_11[i]);

	/// Body origins.
	for (int i = 0; i < 3; ++i) res[0][i + 117] = value<T>(pelvis_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 120] = value<T>(femur_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 123] = value<T>(tibia_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 126] = value<T>(talus_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 129] = value<T>(calcn_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 132] = value<T>(toes_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 135] = value<T>(femur_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 138] = value<T>(tibia_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 141] = value<T>(talus_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 144] = value<T>(calcn_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 147] = value<T>(toes_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 150] = value<T>(torso_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 153] = value<T>(humerus_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 156] = value<T>(ulna_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 159] = value<T>(radius_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 162] = value<T>(hand_r_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 165] = value<T>(humerus_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 168] = value<T>(ulna_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 171] = value<T>(radius_l_or[i]);
	for (int i = 0; i < 3; ++i) res[0][i + 174] = value<T>(hand_l_or[i]);

	return 0;
}

int main() {
	Recorder x[NX];
	Recorder u[NU];
	Recorder tau[NR];
	for (int i = 0; i < NX; ++i) x[i] <<= 0;
	for (int i = 0; i < NU; ++i) u[i] <<= 0;
	const Recorder* Recorder_arg[n_in] = { x,u };
	Recorder* Recorder_res[n_out] = { tau };
	F_generic<Recorder>(Recorder_arg, Recorder_res);
	double res[NR];
	for (int i = 0; i < NR; ++i) Recorder_res[0][i] >>= res[i];
	Recorder::stop_recording();
	return 0;
}
