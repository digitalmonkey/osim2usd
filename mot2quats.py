# For OpenSim support
import math, os, sys, getopt, numpy
import opensim as osim
import csv
from pyquaternion import Quaternion

def saveOutputCSV(outputPath, times, outputNames, outputData):
    file = open(outputPath, "w", newline='')
    writer = csv.writer(file)
    header = []

    for i in range(len(times)):
        row = []

        if i == 0: # Write header
            header.append("time")
            for label in outputNames:
                header.append(label)
            writer.writerow(header)

        row.append(str(times[i]))
        for data in outputData[i]:
            row.append(str(data))
        writer.writerow(row)

    file.close()
def saveMotionCSV(outputPath, bodyNames, times, poseTrajectories):
    file = open(outputPath, "w", newline='')
    writer = csv.writer(file)
    header = []

    for i in range(len(times)):
        row = []  # Row is a list of strings

        if i == 0:
            header.append("time")
        row.append(str(times[i]))
        bodyPoses = poseTrajectories[i]
        for body in range(len(bodyPoses)):
            if i == 0: # Add header info
                header.append(bodyNames[body] + "_x")
                header.append(bodyNames[body] + "_y")
                header.append(bodyNames[body] + "_z")
                header.append(bodyNames[body] + "_qw")
                header.append(bodyNames[body] + "_qx")
                header.append(bodyNames[body] + "_qy")
                header.append(bodyNames[body] + "_qz")

            # Add position
            row.append(str(bodyPoses[body][0][0]))
            row.append(str(bodyPoses[body][0][1]))
            row.append(str(bodyPoses[body][0][2]))

            # Add rotation as quaternion
            # Have to use get() method since can't seem to use subscript to asVec4.
            row.append(str(bodyPoses[body][1].w))
            row.append(str(bodyPoses[body][1].x))
            row.append(str(bodyPoses[body][1].y))
            row.append(str(bodyPoses[body][1].z))




        if i == 0: # Write header before 1st timestamp data
            writer.writerow(header)
        writer.writerow(row)

    file.close()
def mot2quats(motionPath, outputPath, modelPath, optionsDict):

    print(f"Input OpenSim motion path set to: {motionPath}")

    model = osim.Model(modelPath)
    model.initSystem()

    motion = osim.Storage(motionPath)
    motion.setInDegrees(False)

    # Results  states in motionTrajectory are in SimTK:Stage:Instance
    motionTrajectory = osim.StatesTrajectory.createFromStatesStorage(model, motion, True, True)
    print("Trajectory size: ", motionTrajectory.getSize(), " is compatible: ", motionTrajectory.isCompatibleWith(model))
    motionState = motionTrajectory.get(0)
    model.realizePosition(motionState) # Need to do this to query positions.

    bodyNames = []
    workingBodyList = [] # Store bodies to export
    for body in model.getBodyList():
        # Skip body if position information is invalid.
        testPosition = body.getPositionInGround(motionState)
        if math.isnan(testPosition[0]) == False:
            bodyNames.append(body.getName())
            workingBodyList.append(body)


    poseTrajectories = []
    times = []

    referenceBody = None
    if "relativeTo" in optionsDict:
        for body in workingBodyList:
            if body.getName() == optionsDict["relativeTo"]:
                referenceBody = body
                print("Making poses relative to ", referenceBody.getName())

    for i in range(motionTrajectory.getSize()):
        motionState = motionTrajectory.get(i)
        model.realizePosition(motionState)
        times.append(motionState.getTime())

        invReferencePose = None
        if referenceBody != None:

             referenceRotation = referenceBody.getRotationInGround(motionState).convertRotationToQuaternion()
             referenceQuat = Quaternion(referenceRotation.get(0), referenceRotation.get(1), referenceRotation.get(2), referenceRotation.get(3))
             invRotation = referenceQuat.inverse
             invPosition = referenceBody.getPositionInGround(motionState)
             invPosition[0] = -1.0 * invPosition[0]
             invPosition[1] = -1.0 * invPosition[1]
             invPosition[2] = -1.0 * invPosition[2]
             invReferencePose = (invPosition, invRotation)

        # Loop through bodies in model.
        bodyPoses = []
        for body in workingBodyList:
            positionGround = body.getPositionInGround(motionState)
            rotationGround = body.getRotationInGround(motionState).convertRotationToQuaternion()
            bodyQuat = Quaternion(rotationGround.get(0), rotationGround.get(1), rotationGround.get(2), rotationGround.get(3))
            if invReferencePose: # Make pose relative to the reference pose
                 positionGround[0] = positionGround[0] + invReferencePose[0][0]
                 positionGround[1] = positionGround[1] + invReferencePose[0][1]
                 positionGround[2] = positionGround[2] + invReferencePose[0][2]
                 rotationGround = invReferencePose[1] * bodyQuat
            bodyPoses.append((positionGround, rotationGround))
        poseTrajectories.append(bodyPoses)

    if outputPath != None:
        saveMotionCSV(outputPath + "_motion.csv", bodyNames, times, poseTrajectories)

        outputNames = []
        for force in model.getForceSet():
            outputNames.append(force.getName())

        outputData = []

        # Filter out activations only.
        names = model.getStateVariableNames()
        activationLabels = []
        for j in range(names.getSize()):
            if "/activation" in names.get(j):
                activationLabels.append(names.get(j))

        print(f"Found {len(activationLabels)} activation state variables for output.")

        outputNames = []
        for i in range(len(times)):
            sample = []
            state = motionTrajectory.get(i)
            model.realizeVelocity(state)

            for label in activationLabels:
                if i == 0:
                   muscleLabel = label.replace("/activation","")
                   muscleLabel = muscleLabel.replace("/forceset/","")
                   outputNames.append(muscleLabel)
                activation = model.getStateVariableValue(state, label)
                sample.append(str(activation))
            outputData.append(sample)

        saveOutputCSV(outputPath + "_output.csv", times, outputNames, outputData)



    return (bodyNames, times, poseTrajectories)


def main(argv):

    print(f"OpenSim version: {osim.GetVersionAndDate()}")

    sessionPath=""
    modelPath = "./Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim"
    motionPath = "./Motions/kinematics_activations_left_leg_squat_0.mot"
    inputPath = sessionPath + motionPath
    outputPath = os.path.splitext(inputPath)[0]
    optionsDict = dict()

    optionsDict["relativeTo"] = "pelvis"

    opts, args = getopt.getopt(argv,"im:o:",["input=","model=","output="])
    for opt, arg in opts:
        if opt == "-h":
            print("mot2quats.py -i <inputFile> -o <outputFile> -m <modelFile>")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputPath = arg
        elif opt in ("-o", "--output"):
            outputPath = arg
        elif opt in ("-m", "--model"):
            modelPath = arg

    bodyPoseTrajectories = mot2quats(inputPath, outputPath, modelPath, optionsDict)

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])